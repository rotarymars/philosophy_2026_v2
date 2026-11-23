#include "polypack/env.hpp"

#include "polypack/overlap.hpp"
#include "polypack/raster.hpp"
#include "polypack/slide.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace polypack {

std::vector<Rotation> defaultRotations() {
  const Rotation diagonal = Rotation::fromHalfAngleTangent(Rational(5, 12));
  std::vector<Rotation> result;
  for (int k = 0; k < 4; ++k) {
    result.push_back(Rotation::quarterTurns(k));
    result.push_back(Rotation::quarterTurns(k) * diagonal);
  }
  return result;
}

PackingEnv::PackingEnv(EnvConfig config)
    : config_(std::move(config)), rotations_(defaultRotations()) {
  validate(config_.generator);
  if (config_.openBins < 1) throw std::invalid_argument("EnvConfig: openBins must be >= 1");
  if (config_.columns < 1) throw std::invalid_argument("EnvConfig: columns must be >= 1");
  if (config_.rasterSize < 8) throw std::invalid_argument("EnvConfig: rasterSize must be >= 8");
  if (config_.itemsPerEpisode < 1) throw std::invalid_argument("EnvConfig: itemsPerEpisode must be >= 1");
  const std::size_t pixels = static_cast<std::size_t>(config_.rasterSize) * config_.rasterSize;
  itemImages_.assign(rotations_.size() * pixels, 0);
  mask_.assign(static_cast<std::size_t>(actionCount()), 0);
}

void PackingEnv::reset(std::uint64_t seed) {
  rng_.seed(seed);
  open_.clear();
  closed_.clear();
  areaSeen_ = 0;
  placed_ = 0;
  binsUsed_ = 0;
  done_ = false;
  nextItem();
}

const Polygon& PackingEnv::currentItem() const {
  if (!item_) throw std::logic_error("PackingEnv: call reset() first");
  return *item_;
}

void PackingEnv::nextItem() {
  item_ = randomPolygon(rng_, config_.generator);
  areaSeen_ += item_->area();

  const int size = config_.rasterSize;
  const std::size_t pixels = static_cast<std::size_t>(size) * size;
  variants_.clear();
  std::fill(itemImages_.begin(), itemImages_.end(), 0);
  std::vector<std::uint8_t> scratch(pixels);
  for (std::size_t r = 0; r < rotations_.size(); ++r) {
    // Always rotate the original item (never a rotated copy) to keep fractions small.
    Polygon v = item_->rotated(rotations_[r], Point(0, 0));
    v = v.translated(-v.bounds().min);
    std::fill(scratch.begin(), scratch.end(), 0);
    rasterizeInto(v, unitSquare(), size, scratch);
    std::copy(scratch.begin(), scratch.end(), itemImages_.begin() + static_cast<std::ptrdiff_t>(r * pixels));
    variants_.push_back(std::move(v));
  }
  computeMask();
}

Polygon PackingEnv::entryPlacement(int rotation, int column) const {
  const Polygon& v = variants_[static_cast<std::size_t>(rotation)];
  const Rational& width = v.bounds().max.x;
  const Rational& height = v.bounds().max.y;
  const Rational x = config_.columns == 1
                         ? Rational(0)
                         : Rational((1 - width) * column / (config_.columns - 1));
  return v.translated(Point(x, Rational(1 - height)));
}

void PackingEnv::computeMask() {
  std::fill(mask_.begin(), mask_.end(), 0);
  const Box unit = unitSquare();
  for (int r = 0; r < rotations(); ++r) {
    for (int c = 0; c < columns(); ++c) {
      const Polygon entry = entryPlacement(r, c);
      if (!isContainedIn(entry, unit)) continue;  // cannot happen for valid generator configs
      for (std::size_t s = 0; s < open_.size(); ++s) {
        if (open_[s].bin.canPlace(entry)) {
          mask_[static_cast<std::size_t>(encodeAction(static_cast<int>(s), r, c))] = 1;
        }
      }
    }
    // In an empty bin every column settles to the same bottom-left spot, so
    // only column 0 is offered: one action per rotation instead of many
    // duplicates (which would also make a fresh policy open bins far too often).
    mask_[static_cast<std::size_t>(encodeAction(config_.openBins, r, 0))] = 1;
  }
}

bool PackingEnv::isLegal(int action) const {
  return !done_ && action >= 0 && action < actionCount() &&
         mask_[static_cast<std::size_t>(action)] != 0;
}

Polygon PackingEnv::simulate(int action) const {
  if (!isLegal(action)) throw std::invalid_argument("PackingEnv: illegal action " + std::to_string(action));
  const int column = action % columns();
  const int rotation = (action / columns()) % rotations();
  const int slot = action / (columns() * rotations());
  static const std::vector<Polygon> kNoObstacles;
  const std::vector<Polygon>& obstacles =
      slot == config_.openBins ? kNoObstacles : open_[static_cast<std::size_t>(slot)].bin.items();
  return settleBottomLeft(entryPlacement(rotation, column), obstacles, unitSquare());
}

StepResult PackingEnv::step(int action) {
  const Polygon placement = simulate(action);  // validates the action
  const int slot = action / (columns() * rotations());

  StepResult result;
  std::size_t binIndex = static_cast<std::size_t>(slot);
  if (slot == config_.openBins) {
    const std::size_t pixels = static_cast<std::size_t>(config_.rasterSize) * config_.rasterSize;
    open_.push_back(OpenBin{Bin(), std::vector<std::uint8_t>(pixels, 0)});
    ++binsUsed_;
    result.reward = -1.0f;
    if (static_cast<int>(open_.size()) > config_.openBins) {
      closed_.push_back(std::move(open_.front().bin));
      open_.pop_front();
    }
    binIndex = open_.size() - 1;
  }

  OpenBin& target = open_[binIndex];
  if (!target.bin.tryPlace(placement)) {
    // Settling preserves validity exactly, so this would be a library bug.
    throw std::logic_error("PackingEnv: settled placement was rejected (bug)");
  }
  rasterizeInto(placement, unitSquare(), config_.rasterSize, target.image);

  ++placed_;
  if (placed_ >= config_.itemsPerEpisode) {
    done_ = true;
    std::fill(mask_.begin(), mask_.end(), 0);
  } else {
    nextItem();
  }
  result.done = done_;
  return result;
}

void PackingEnv::observe(std::uint8_t* binImages, std::uint8_t* itemImages, float* slotFeatures,
                         float* globalFeatures, std::uint8_t* mask) const {
  const std::size_t pixels = static_cast<std::size_t>(config_.rasterSize) * config_.rasterSize;
  for (int s = 0; s < slots(); ++s) {
    std::uint8_t* dst = binImages + static_cast<std::size_t>(s) * pixels;
    float* f = slotFeatures + static_cast<std::size_t>(s) * kSlotFeatures;
    const bool isOpen = static_cast<std::size_t>(s) < open_.size();
    if (isOpen) {
      std::memcpy(dst, open_[static_cast<std::size_t>(s)].image.data(), pixels);
    } else {
      std::memset(dst, 0, pixels);
    }
    f[0] = isOpen ? 1.0f : 0.0f;
    f[1] = s == config_.openBins ? 1.0f : 0.0f;
    f[2] = isOpen ? static_cast<float>(toDouble(open_[static_cast<std::size_t>(s)].bin.fillRatio())) : 0.0f;
  }
  std::memcpy(itemImages, itemImages_.data(), itemImages_.size());
  globalFeatures[0] = item_ ? static_cast<float>(toDouble(item_->area()) * 10.0) : 0.0f;
  globalFeatures[1] = static_cast<float>(placed_) / static_cast<float>(config_.itemsPerEpisode);
  globalFeatures[2] = static_cast<float>(open_.size()) / static_cast<float>(config_.openBins);
  std::memcpy(mask, mask_.data(), mask_.size());
}

int PackingEnv::areaLowerBound() const {
  mpz_class bound;
  mpz_cdiv_q(bound.get_mpz_t(), areaSeen_.get_num_mpz_t(), areaSeen_.get_den_mpz_t());
  return static_cast<int>(bound.get_si());
}

std::vector<Bin> PackingEnv::allBins() const {
  std::vector<Bin> bins = closed_;
  for (const OpenBin& b : open_) bins.push_back(b.bin);
  return bins;
}

int firstFitAction(const PackingEnv& env) {
  const std::vector<std::uint8_t>& mask = env.mask();
  for (int a = 0; a < env.actionCount(); ++a) {
    if (mask[static_cast<std::size_t>(a)]) return a;
  }
  throw std::logic_error("firstFitAction: no legal action");
}

int greedyLowestAction(const PackingEnv& env) {
  const std::vector<std::uint8_t>& mask = env.mask();
  const int perSlot = env.rotations() * env.columns();
  for (int s = 0; s < env.slots(); ++s) {
    int best = -1;
    std::optional<Polygon> bestPlacement;
    for (int a = s * perSlot; a < (s + 1) * perSlot; ++a) {
      if (!mask[static_cast<std::size_t>(a)]) continue;
      Polygon p = env.simulate(a);
      if (!bestPlacement || p.bounds().max.y < bestPlacement->bounds().max.y ||
          (p.bounds().max.y == bestPlacement->bounds().max.y &&
           p.bounds().max.x < bestPlacement->bounds().max.x)) {
        best = a;
        bestPlacement = std::move(p);
      }
    }
    if (best >= 0) return best;
  }
  throw std::logic_error("greedyLowestAction: no legal action");
}

}  // namespace polypack
