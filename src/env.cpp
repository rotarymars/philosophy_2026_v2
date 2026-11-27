#include "polypack/env.hpp"

#include "polypack/overlap.hpp"
#include "polypack/raster.hpp"
#include "polypack/slide.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

namespace polypack {
namespace {

long floorOf(const Rational& q) {
  mpz_class f;
  mpz_fdiv_q(f.get_mpz_t(), q.get_num_mpz_t(), q.get_den_mpz_t());
  return f.get_si();
}

long ceilOf(const Rational& q) {
  mpz_class c;
  mpz_cdiv_q(c.get_mpz_t(), q.get_num_mpz_t(), q.get_den_mpz_t());
  return c.get_si();
}

// Integers strictly between a and b: [floor(a) + 1, ceil(b) - 1].
std::pair<long, long> openIntegerRange(const Rational& a, const Rational& b) {
  return {floorOf(a) + 1, ceilOf(b) - 1};
}

}  // namespace

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
  if (config_.grid < 1 || config_.grid > 256) throw std::invalid_argument("EnvConfig: grid must be in 1..256");
  if (config_.rasterSize < 8) throw std::invalid_argument("EnvConfig: rasterSize must be >= 8");
  if (config_.itemsPerEpisode < 1) throw std::invalid_argument("EnvConfig: itemsPerEpisode must be >= 1");
  if (config_.minItemsPerEpisode < 0) throw std::invalid_argument("EnvConfig: minItemsPerEpisode must be >= 0");
  if (config_.maskThreads < 1) throw std::invalid_argument("EnvConfig: maskThreads must be >= 1");
  for (int g = 0; g < config_.grid; ++g) cellOffset_.emplace_back(Rational(g, config_.grid));
  const std::size_t pixels = static_cast<std::size_t>(config_.rasterSize) * config_.rasterSize;
  itemImages_.assign(rotations_.size() * pixels, 0);
  if (!config_.lazyMask) mask_.assign(static_cast<std::size_t>(actionCount()), 0);
}

void PackingEnv::reset(std::uint64_t seed) { startEpisode(episodeItems(config_, seed)); }

void PackingEnv::resetWithItems(std::vector<Polygon> items) {
  if (items.empty()) throw std::invalid_argument("PackingEnv::resetWithItems: no items");
  startEpisode(std::move(items));
}

void PackingEnv::startEpisode(std::vector<Polygon> items) {
  items_ = std::move(items);
  episodeLength_ = static_cast<int>(items_.size());
  areaAfter_.assign(items_.size(), 0.0);
  for (std::size_t i = items_.size(); i-- > 1;) areaAfter_[i - 1] = areaAfter_[i] + toDouble(items_[i].area());
  open_.clear();
  closed_.clear();
  areaSeen_ = 0;
  placed_ = 0;
  binsUsed_ = 0;
  done_ = false;
  nextItem();
}

int episodeLength(const EnvConfig& config, std::uint64_t seed) {
  const int lo = config.minItemsPerEpisode;
  const int hi = config.itemsPerEpisode;
  if (lo <= 0 || lo >= hi) return hi;
  // A generator of its own, so the polygons themselves do not depend on the
  // count range. Plain modulo keeps the result identical on every platform.
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
  return lo + static_cast<int>(rng() % static_cast<std::uint64_t>(hi - lo + 1));
}

std::vector<Polygon> episodeItems(const EnvConfig& config, std::uint64_t seed) {
  // Must match PackingEnv::reset + nextItem: one randomPolygon call per item.
  std::mt19937_64 rng(seed);
  const int count = episodeLength(config, seed);
  std::vector<Polygon> items;
  items.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) items.push_back(randomPolygon(rng, config.generator));
  return items;
}

const Polygon& PackingEnv::currentItem() const {
  if (!item_) throw std::logic_error("PackingEnv: call reset() first");
  return *item_;
}

Action PackingEnv::decodeAction(int action) const {
  Action a;
  a.gx = action % grid();
  a.gy = (action / grid()) % grid();
  a.rotation = (action / positions()) % rotations();
  a.slot = action / (positions() * rotations());
  return a;
}

PackingEnv::Diagonals PackingEnv::diagonalsOf(const Polygon& p) {
  Diagonals d;
  bool first = true;
  for (const Point& v : p.vertices()) {
    Rational s = v.x + v.y;
    Rational t = v.x - v.y;
    if (first || s < d.minSum) d.minSum = s;
    if (first || s > d.maxSum) d.maxSum = s;
    if (first || t < d.minDiff) d.minDiff = t;
    if (first || t > d.maxDiff) d.maxDiff = t;
    first = false;
  }
  return d;
}

int PackingEnv::lastFittingCell(const Rational& extent) const {
  const Rational room = 1 - extent;
  if (room < 0) return -1;
  return static_cast<int>(std::min<long>(config_.grid - 1, floorOf(Rational(room * config_.grid))));
}

void PackingEnv::nextItem() {
  item_ = items_[static_cast<std::size_t>(placed_)];
  areaSeen_ += item_->area();

  const int size = config_.rasterSize;
  const std::size_t pixels = static_cast<std::size_t>(size) * size;
  variants_.clear();
  variantDiag_.clear();
  std::fill(itemImages_.begin(), itemImages_.end(), 0);
  std::vector<std::uint8_t> scratch(pixels);
  for (std::size_t r = 0; r < rotations_.size(); ++r) {
    // Always rotate the original item (never a rotated copy) to keep fractions small.
    Polygon v = item_->rotated(rotations_[r], Point(0, 0));
    v = v.translated(-v.bounds().min);
    std::fill(scratch.begin(), scratch.end(), 0);
    rasterizeInto(v, unitSquare(), size, scratch);
    std::copy(scratch.begin(), scratch.end(), itemImages_.begin() + static_cast<std::ptrdiff_t>(r * pixels));
    variantDiag_.push_back(diagonalsOf(v));
    variants_.push_back(std::move(v));
  }
  if (!config_.lazyMask) computeMask();
}

// Exact legality of every grid spot for one (open bin, rotation) pair.
//
// Speed-ups that keep the answer exact: an item can only overlap the
// candidate if their bounding boxes AND their extents along x+y and x-y all
// overlap with positive length. For each item those four conditions become
// integer ranges of gx, gy, gx+gy and gx-gy (computed with exact rational
// floor/ceil), so most (spot, item) pairs are ruled out by integer compares.
// Every remaining pair gets the full exact interiorsOverlap test.
int PackingEnv::scanSpots(int slot, int rotation, std::uint8_t* out) const {
  const Polygon& v = variants_[static_cast<std::size_t>(rotation)];
  const Diagonals& vd = variantDiag_[static_cast<std::size_t>(rotation)];
  const Rational& w = v.bounds().max.x;  // the variant's bounding box starts at (0, 0)
  const Rational& h = v.bounds().max.y;
  const int gxMax = lastFittingCell(w);
  const int gyMax = lastFittingCell(h);
  if (gxMax < 0 || gyMax < 0) return -1;

  const OpenBin& ob = open_[static_cast<std::size_t>(slot)];
  // Exact shortcut: if the item's area exceeds the bin's free area, no spot
  // can be legal (every spot stays 0, as the full check would find).
  if (ob.bin.usedArea() + item_->area() > Rational(1)) return -1;
  const std::vector<Polygon>& items = ob.bin.items();
  const Rational g(config_.grid);

  struct Near {
    long xLo, xHi, yLo, yHi, sLo, sHi, dLo, dHi;
    std::size_t index;
  };
  std::vector<Near> near;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const Box& b = items[i].bounds();
    const Diagonals& d = ob.diagonals[i];
    Near n{};
    std::tie(n.xLo, n.xHi) = openIntegerRange(Rational(g * (b.min.x - w)), Rational(g * b.max.x));
    std::tie(n.yLo, n.yHi) = openIntegerRange(Rational(g * (b.min.y - h)), Rational(g * b.max.y));
    std::tie(n.sLo, n.sHi) = openIntegerRange(Rational(g * (d.minSum - vd.maxSum)), Rational(g * (d.maxSum - vd.minSum)));
    std::tie(n.dLo, n.dHi) = openIntegerRange(Rational(g * (d.minDiff - vd.maxDiff)), Rational(g * (d.maxDiff - vd.minDiff)));
    if (n.xLo > std::min<long>(n.xHi, gxMax) || n.xHi < 0) continue;
    if (n.yLo > std::min<long>(n.yHi, gyMax) || n.yHi < 0) continue;
    if (n.sLo > n.sHi || n.dLo > n.dHi) continue;
    n.index = i;
    near.push_back(n);
  }

  std::vector<const Near*> row;
  std::vector<std::size_t> hits;
  int first = -1;
  for (int gy = 0; gy <= gyMax; ++gy) {
    row.clear();
    for (const Near& n : near) {
      if (gy >= n.yLo && gy <= n.yHi) row.push_back(&n);
    }
    for (int gx = 0; gx <= gxMax; ++gx) {
      hits.clear();
      for (const Near* n : row) {
        if (gx >= n->xLo && gx <= n->xHi && gx + gy >= n->sLo && gx + gy <= n->sHi &&
            gx - gy >= n->dLo && gx - gy <= n->dHi) {
          hits.push_back(n->index);
        }
      }
      bool legal = true;
      if (!hits.empty()) {
        const Point offset(cellOffset_[static_cast<std::size_t>(gx)], cellOffset_[static_cast<std::size_t>(gy)]);
        const Polygon placed = v.translated(offset);
        for (std::size_t k : hits) {
          if (interiorsOverlap(items[k], placed)) {
            legal = false;
            break;
          }
        }
      }
      const int action = encodeAction(slot, rotation, gx, gy);
      if (legal && first < 0) {
        first = action;
        if (out == nullptr) return first;
      }
      if (out != nullptr) out[static_cast<std::size_t>(action)] = legal ? 1 : 0;
    }
  }
  return first;
}

bool PackingEnv::newBinLegal(int rotation, int gx, int gy) const {
  // Empty bin: legal wherever the item lies inside. With settling only (0, 0)
  // is offered, since every spot settles into the same corner.
  const Polygon& v = variants_[static_cast<std::size_t>(rotation)];
  const int gxMax = config_.settle ? std::min(0, lastFittingCell(v.bounds().max.x)) : lastFittingCell(v.bounds().max.x);
  const int gyMax = config_.settle ? std::min(0, lastFittingCell(v.bounds().max.y)) : lastFittingCell(v.bounds().max.y);
  return gx >= 0 && gy >= 0 && gx <= gxMax && gy <= gyMax;
}

int PackingEnv::firstLegal(int slot, int rotation) const {
  if (done_) return -1;
  if (!config_.lazyMask) {
    const int base = encodeAction(slot, rotation, 0, 0);
    for (int k = 0; k < positions(); ++k) {
      if (mask_[static_cast<std::size_t>(base + k)]) return base + k;
    }
    return -1;
  }
  if (slot == config_.openBins) {
    for (int gy = 0; gy < grid(); ++gy) {
      for (int gx = 0; gx < grid(); ++gx) {
        if (newBinLegal(rotation, gx, gy)) return encodeAction(slot, rotation, gx, gy);
      }
    }
    return -1;
  }
  if (slot < 0 || static_cast<std::size_t>(slot) >= open_.size()) return -1;
  return scanSpots(slot, rotation, nullptr);
}

void PackingEnv::computeMask() {
  std::fill(mask_.begin(), mask_.end(), 0);

  // New bin: empty, so every spot where the item fits inside is legal. With
  // settling they all end in the same corner, so only (0, 0) is offered: one
  // action per rotation instead of many duplicates (which would also make a
  // fresh policy open bins far too often).
  for (int r = 0; r < rotations(); ++r) {
    for (int gy = 0; gy < grid(); ++gy) {
      for (int gx = 0; gx < grid(); ++gx) {
        if (newBinLegal(r, gx, gy)) mask_[static_cast<std::size_t>(encodeAction(config_.openBins, r, gx, gy))] = 1;
      }
    }
  }

  const int tasks = static_cast<int>(open_.size()) * rotations();
  const auto run = [&](int t) { scanSpots(t / rotations(), t % rotations(), mask_.data()); };
  const int workers = std::min(config_.maskThreads, tasks);
  if (workers <= 1) {
    for (int t = 0; t < tasks; ++t) run(t);
    return;
  }
  std::atomic<int> next{0};
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
  std::vector<std::thread> pool;
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      try {
        for (int t = next++; t < tasks; t = next++) run(t);
      } catch (...) {
        errors[static_cast<std::size_t>(w)] = std::current_exception();
      }
    });
  }
  for (std::thread& t : pool) t.join();
  for (const std::exception_ptr& e : errors) {
    if (e) std::rethrow_exception(e);
  }
}

bool PackingEnv::isLegal(int action) const {
  if (done_ || action < 0 || action >= actionCount()) return false;
  if (!config_.lazyMask) return mask_[static_cast<std::size_t>(action)] != 0;
  // Lazy: the same exact test, for this one spot.
  const Action a = decodeAction(action);
  if (a.slot == config_.openBins) return newBinLegal(a.rotation, a.gx, a.gy);
  if (static_cast<std::size_t>(a.slot) >= open_.size()) return false;
  const Polygon& v = variants_[static_cast<std::size_t>(a.rotation)];
  if (a.gx > lastFittingCell(v.bounds().max.x) || a.gy > lastFittingCell(v.bounds().max.y)) return false;
  const Polygon placed = v.translated(Point(cellOffset_[static_cast<std::size_t>(a.gx)],
                                            cellOffset_[static_cast<std::size_t>(a.gy)]));
  return open_[static_cast<std::size_t>(a.slot)].bin.canPlace(placed);
}

Polygon PackingEnv::anchorPlacement(int action) const {
  if (action < 0 || action >= actionCount()) {
    throw std::invalid_argument("PackingEnv: action out of range " + std::to_string(action));
  }
  const Action a = decodeAction(action);
  return variants_[static_cast<std::size_t>(a.rotation)].translated(
      Point(cellOffset_[static_cast<std::size_t>(a.gx)], cellOffset_[static_cast<std::size_t>(a.gy)]));
}

const std::vector<Polygon>& PackingEnv::slotItems(int slot) const {
  static const std::vector<Polygon> kNone;
  if (slot >= 0 && static_cast<std::size_t>(slot) < open_.size() && slot != config_.openBins) {
    return open_[static_cast<std::size_t>(slot)].bin.items();
  }
  return kNone;
}

Polygon PackingEnv::simulate(int action) const {
  if (!isLegal(action)) throw std::invalid_argument("PackingEnv: illegal action " + std::to_string(action));
  Polygon start = anchorPlacement(action);
  if (!config_.settle) return start;
  return settleBottomLeft(start, slotItems(decodeAction(action).slot), unitSquare());
}

StepResult PackingEnv::step(int action) {
  const Polygon placement = simulate(action);  // validates the action
  const int slot = decodeAction(action).slot;

  StepResult result;
  std::size_t binIndex = static_cast<std::size_t>(slot);
  if (slot == config_.openBins) {
    const std::size_t pixels = static_cast<std::size_t>(config_.rasterSize) * config_.rasterSize;
    open_.push_back(OpenBin{Bin(), std::vector<std::uint8_t>(pixels, 0), {}});
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
    // The mask and settling are exact, so this would be a library bug.
    throw std::logic_error("PackingEnv: placement was rejected (bug)");
  }
  target.diagonals.push_back(diagonalsOf(placement));
  rasterizeInto(placement, unitSquare(), config_.rasterSize, target.image);

  ++placed_;
  if (placed_ >= episodeLength_) {
    done_ = true;
    std::fill(mask_.begin(), mask_.end(), 0);
  } else {
    nextItem();
  }
  result.done = done_;
  return result;
}

void PackingEnv::observe(std::uint8_t* binImages, std::uint8_t* itemImages, std::uint8_t* contactImages,
                         float* slotFeatures, float* globalFeatures, std::uint8_t* mask) const {
  if (config_.lazyMask) throw std::logic_error("PackingEnv::observe: not available with lazyMask");
  const int size = config_.rasterSize;
  const std::size_t pixels = static_cast<std::size_t>(size) * size;
  for (int s = 0; s < slots(); ++s) {
    std::uint8_t* dst = binImages + static_cast<std::size_t>(s) * pixels;
    float* f = slotFeatures + static_cast<std::size_t>(s) * kSlotFeatures;
    const bool isOpen = s != config_.openBins && static_cast<std::size_t>(s) < open_.size();
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

  // Contact hints. The item image has its lower-left pixel at (size-1, 0), so
  // item pixel (r, c) lands on bin pixel (row + r - (size-1), col + c).
  std::vector<std::pair<int, int>> ring;
  for (int r = 0; r < rotations(); ++r) {
    const std::uint8_t* item = itemImages_.data() + static_cast<std::size_t>(r) * pixels;
    const auto filled = [&](int y, int x) {
      return y >= 0 && y < size && x >= 0 && x < size && item[static_cast<std::size_t>(y) * size + x] > 0;
    };
    ring.clear();
    for (int y = -1; y <= size; ++y) {
      for (int x = -1; x <= size; ++x) {
        if (filled(y, x)) continue;
        bool touches = false;
        for (int dy = -1; dy <= 1 && !touches; ++dy) {
          for (int dx = -1; dx <= 1 && !touches; ++dx) touches = filled(y + dy, x + dx);
        }
        if (touches) ring.emplace_back(y - (size - 1), x);
      }
    }
    for (int s = 0; s < slots(); ++s) {
      const std::uint8_t* bin = binImages + static_cast<std::size_t>(s) * pixels;
      std::uint8_t* out = contactImages + (static_cast<std::size_t>(s) * rotations() + r) * pixels;
      if (ring.empty()) {
        std::memset(out, 0, pixels);
        continue;
      }
      for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
          long sum = 0;
          for (const auto& [dy, dx] : ring) {
            const int y = row + dy;
            const int x = col + dx;
            sum += (y < 0 || y >= size || x < 0 || x >= size) ? 255 : bin[static_cast<std::size_t>(y) * size + x];
          }
          out[static_cast<std::size_t>(row) * size + col] = static_cast<std::uint8_t>(sum / static_cast<long>(ring.size()));
        }
      }
    }
  }

  globalFeatures[0] = item_ ? static_cast<float>(toDouble(item_->area()) * 10.0) : 0.0f;
  // Divided by the largest possible count, not this episode's: the player must
  // not learn how many items are left.
  globalFeatures[1] = static_cast<float>(placed_) / static_cast<float>(config_.itemsPerEpisode);
  globalFeatures[2] = static_cast<float>(open_.size()) / static_cast<float>(config_.openBins);
  // Items so far, counting the current one. Given that many have arrived,
  // the count is uniform on [max(lo, seen), hi].
  const int hi = config_.itemsPerEpisode;
  const int lo = config_.minItemsPerEpisode <= 0 || config_.minItemsPerEpisode >= hi ? hi : config_.minItemsPerEpisode;
  const int seen = placed_ + 1;
  const int from = std::min(hi, std::max(lo, seen));
  const double expectedLeft = std::max(0.0, 0.5 * (from + hi) - seen);
  globalFeatures[3] = done_ ? 0.0f : static_cast<float>(expectedLeft / hi);
  globalFeatures[4] = done_ || seen < lo ? 0.0f : static_cast<float>(1.0 / (hi - from + 1));
  std::memcpy(mask, mask_.data(), mask_.size());
}

void PackingEnv::criticFeatures(float* out) const {
  if (done_) {
    out[0] = out[1] = 0.0f;
    return;
  }
  out[0] = static_cast<float>(episodeLength_ - placed_ - 1) / static_cast<float>(config_.itemsPerEpisode);
  out[1] = static_cast<float>(areaAfter_[static_cast<std::size_t>(placed_)]);
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

int greedyBottomLeftAction(const PackingEnv& env) {
  for (int s = 0; s < env.slots(); ++s) {
    int best = -1;
    std::optional<Polygon> bestPlacement;
    for (int r = 0; r < env.rotations(); ++r) {
      const int first = env.firstLegal(s, r);
      if (first < 0) continue;
      Polygon p = env.simulate(first);
      if (!bestPlacement || p.bounds().max.y < bestPlacement->bounds().max.y ||
          (p.bounds().max.y == bestPlacement->bounds().max.y &&
           p.bounds().max.x < bestPlacement->bounds().max.x)) {
        best = first;
        bestPlacement = std::move(p);
      }
    }
    if (best >= 0) return best;
  }
  throw std::logic_error("greedyBottomLeftAction: no legal action");
}

}  // namespace polypack
