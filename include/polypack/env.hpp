#pragma once

// The online 2D bin-packing environment used for training.
//
// Episode: `itemsPerEpisode` random polygons arrive one at a time. Each must
// be placed immediately and permanently.
//
// Bins: at most `openBins` bins stay open (the bounded-space variant of
// online bin packing). Opening a new bin while that many are open closes the
// oldest one for good.
//
// Action = (slot, rotation, gx, gy)  -- free placement:
//   * slot      0..openBins-1 are the open bins, oldest first;
//               slot openBins means "open a new bin".
//   * rotation  one of the exact rotations from defaultRotations().
//   * gx, gy    a cell of a grid x grid placement grid. The rotated item is put
//               directly at that spot, with the lower-left corner of its
//               bounding box at (gx / grid, gy / grid). Nothing has to be
//               reachable from the top: any spot that is free can be used,
//               including holes under overhangs.
//   * If `settle` is on (the default), the item then slides down and left
//     from that spot until it rests (exact bottom-left settling), so pieces
//     end up exactly touching instead of leaving grid-sized gaps.
// An action is legal (mask = 1) iff the item fits inside the bin at that spot
// without overlapping anything, checked exactly. With `settle` on, the
// new-bin slot only offers gx = gy = 0, since every spot in an empty bin
// settles into the same corner.
//
// Action index = ((slot * rotations + rotation) * grid + gy) * grid + gx,
// so lower rows come first and, within a row, the left-most cell.
//
// Reward: -1 each time a bin is opened, so the return is -(bins used).

#include "polypack/bin.hpp"
#include "polypack/generator.hpp"
#include "polypack/polygon.hpp"
#include "polypack/rotation.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <vector>

namespace polypack {

struct EnvConfig {
  int openBins = 2;
  int grid = 16;          // placement grid: grid x grid spots per bin and rotation
  bool settle = true;     // slide down/left from the chosen spot after placing
  int rasterSize = 32;
  int itemsPerEpisode = 60;
  int maskThreads = 1;    // threads for the legality mask (keep 1 when envs already run in parallel)
  GeneratorConfig generator;
};

// 8 exact rotations: k * 90 degrees, and k * 90 degrees + 2*atan(5/12)
// (about 45.24 degrees, the 5-12-13 rotation).
std::vector<Rotation> defaultRotations();

struct StepResult {
  float reward = 0.0f;
  bool done = false;
};

struct Action {
  int slot = 0;
  int rotation = 0;
  int gx = 0;
  int gy = 0;
};

class PackingEnv {
 public:
  static constexpr int kSlotFeatures = 3;    // is open bin, is new-bin slot, fill ratio
  static constexpr int kGlobalFeatures = 3;  // item area * 10, episode progress, open-bin load

  explicit PackingEnv(EnvConfig config);

  // Starts a new episode; the item sequence is fully determined by `seed`
  // (it is exactly episodeItems(config, seed)).
  void reset(std::uint64_t seed);
  // Starts a new episode whose items arrive in exactly this order (used by
  // the offline packer). Requires items.size() == itemsPerEpisode.
  void resetWithItems(std::vector<Polygon> items);

  const EnvConfig& config() const noexcept { return config_; }
  int slots() const noexcept { return config_.openBins + 1; }
  int rotations() const noexcept { return static_cast<int>(rotations_.size()); }
  int grid() const noexcept { return config_.grid; }
  int positions() const noexcept { return config_.grid * config_.grid; }
  int actionCount() const noexcept { return slots() * rotations() * positions(); }
  int rasterSize() const noexcept { return config_.rasterSize; }

  int encodeAction(int slot, int rotation, int gx, int gy) const {
    return ((slot * rotations() + rotation) * grid() + gy) * grid() + gx;
  }
  int encodeAction(const Action& a) const { return encodeAction(a.slot, a.rotation, a.gx, a.gy); }
  Action decodeAction(int action) const;

  // Writes the observation into caller-provided buffers of sizes
  //   binImages      slots * R * R              (R = rasterSize; 0..255 coverage, row 0 = top)
  //   itemImages     rotations * R * R          (item drawn at bin scale, bottom-left anchored)
  //   contactImages  slots * rotations * R * R  (see below)
  //   slotFeatures   slots * kSlotFeatures
  //   globalFeatures kGlobalFeatures
  //   mask           actionCount                (1 = legal)
  // contactImages[s][r][row][col] estimates, from the pictures, how much of
  // the ring of pixels around the item (rotation r) would touch filled pixels
  // or walls if the item's lower-left pixel sat at (row, col) in bin s
  // (0 = nothing nearby, 255 = surrounded). It is a hint for the network
  // only; legality always comes from the exact mask.
  void observe(std::uint8_t* binImages, std::uint8_t* itemImages, std::uint8_t* contactImages,
               float* slotFeatures, float* globalFeatures, std::uint8_t* mask) const;

  const std::vector<std::uint8_t>& mask() const noexcept { return mask_; }
  bool isLegal(int action) const;

  // The current item, rotated and put at the action's grid spot, before any
  // settling. Defined for every action index (it may lie outside the bin).
  Polygon anchorPlacement(int action) const;
  // Where `action` finally puts the current item (exact). Requires isLegal(action).
  Polygon simulate(int action) const;
  // Items already in the bin behind `slot` (empty for the new-bin slot).
  const std::vector<Polygon>& slotItems(int slot) const;

  // Applies a legal action. Throws std::invalid_argument for illegal actions.
  StepResult step(int action);

  bool done() const noexcept { return done_; }
  int itemsPlaced() const noexcept { return placed_; }
  int binsUsed() const noexcept { return binsUsed_; }
  const Polygon& currentItem() const;
  // Exact total area of all items seen so far, and ceil() of it: no packing
  // can use fewer bins than that.
  const Rational& itemAreaSoFar() const noexcept { return areaSeen_; }
  int areaLowerBound() const;
  // Every bin of the episode (closed and open) in opening order.
  std::vector<Bin> allBins() const;

 private:
  // Exact extents of a polygon along x+y and x-y (with the bounding box this
  // is an octagon around it); used to skip far-away items without losing
  // exactness.
  struct Diagonals {
    Rational minSum, maxSum, minDiff, maxDiff;
  };
  struct OpenBin {
    Bin bin;
    std::vector<std::uint8_t> image;
    std::vector<Diagonals> diagonals;  // one per item, same order
  };

  static Diagonals diagonalsOf(const Polygon& p);
  void nextItem();
  void computeMask();
  void computeMaskFor(int slot, int rotation);
  // Largest g >= 0 with g / grid + extent <= 1, or -1 if none.
  int lastFittingCell(const Rational& extent) const;

  EnvConfig config_;
  std::vector<Rotation> rotations_;
  std::vector<Rational> cellOffset_;  // g / grid for g = 0..grid-1
  std::mt19937_64 rng_;
  std::vector<Polygon> fixedItems_;  // non-empty: items come from here, not from rng_
  std::deque<OpenBin> open_;
  std::vector<Bin> closed_;
  std::optional<Polygon> item_;
  std::vector<Polygon> variants_;       // item under each rotation, bounding box at (0, 0)
  std::vector<Diagonals> variantDiag_;  // diagonals of each variant
  std::vector<std::uint8_t> itemImages_;
  std::vector<std::uint8_t> mask_;
  Rational areaSeen_;
  int placed_ = 0;
  int binsUsed_ = 0;
  bool done_ = true;
};

// The items an episode with this seed delivers, in arrival order.
std::vector<Polygon> episodeItems(const EnvConfig& config, std::uint64_t seed);

// Baselines (no learning), useful as references.
// First legal action in index order: oldest bin first, rotation 0 first, then
// the lowest, left-most free grid spot.
int firstFitAction(const PackingEnv& env);
// Bottom-left fill: in the oldest open bin that can take the item, try every
// rotation at its lowest, left-most free grid spot, settle it, and keep the
// one whose top edge ends lowest (then the left-most).
int greedyBottomLeftAction(const PackingEnv& env);

}  // namespace polypack
