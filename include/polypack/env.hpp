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
// Action = (slot, rotation, column):
//   * slot     0..openBins-1 are the open bins, oldest first;
//              slot openBins means "open a new bin".
//   * rotation one of the exact rotations from defaultRotations().
//   * column   the item enters from the top of the bin with its bounding box
//              at x = (1 - width) * column / (columns - 1). It then falls down
//              and slides left until it rests (exact bottom-left settling).
// An action is legal (mask = 1) iff the entry position at the top of the bin
// is free, checked exactly. For the new-bin slot only column 0 is legal, since
// every column settles to the same spot in an empty bin.
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
  int columns = 16;
  int rasterSize = 32;
  int itemsPerEpisode = 60;
  GeneratorConfig generator;
};

// 8 exact rotations: k * 90 degrees, and k * 90 degrees + 2*atan(5/12)
// (about 45.24 degrees, the 5-12-13 rotation).
std::vector<Rotation> defaultRotations();

struct StepResult {
  float reward = 0.0f;
  bool done = false;
};

class PackingEnv {
 public:
  static constexpr int kSlotFeatures = 3;    // is open bin, is new-bin slot, fill ratio
  static constexpr int kGlobalFeatures = 3;  // item area * 10, episode progress, open-bin load

  explicit PackingEnv(EnvConfig config);

  // Starts a new episode; the item sequence is fully determined by `seed`.
  void reset(std::uint64_t seed);

  const EnvConfig& config() const noexcept { return config_; }
  int slots() const noexcept { return config_.openBins + 1; }
  int rotations() const noexcept { return static_cast<int>(rotations_.size()); }
  int columns() const noexcept { return config_.columns; }
  int actionCount() const noexcept { return slots() * rotations() * columns(); }
  int rasterSize() const noexcept { return config_.rasterSize; }

  int encodeAction(int slot, int rotation, int column) const {
    return (slot * rotations() + rotation) * columns() + column;
  }

  // Writes the observation into caller-provided buffers of sizes
  //   binImages      slots * R * R        (R = rasterSize; 0..255 coverage)
  //   itemImages     rotations * R * R    (item drawn at bin scale, bottom-left anchored)
  //   slotFeatures   slots * kSlotFeatures
  //   globalFeatures kGlobalFeatures
  //   mask           actionCount          (1 = legal)
  void observe(std::uint8_t* binImages, std::uint8_t* itemImages, float* slotFeatures,
               float* globalFeatures, std::uint8_t* mask) const;

  const std::vector<std::uint8_t>& mask() const noexcept { return mask_; }
  bool isLegal(int action) const;

  // Where `action` would put the current item (exact). Requires isLegal(action).
  Polygon simulate(int action) const;

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
  struct OpenBin {
    Bin bin;
    std::vector<std::uint8_t> image;
  };

  void nextItem();
  void computeMask();
  Polygon entryPlacement(int rotation, int column) const;

  EnvConfig config_;
  std::vector<Rotation> rotations_;
  std::mt19937_64 rng_;
  std::deque<OpenBin> open_;
  std::vector<Bin> closed_;
  std::optional<Polygon> item_;
  std::vector<Polygon> variants_;  // item under each rotation, bounding box at (0, 0)
  std::vector<std::uint8_t> itemImages_;
  std::vector<std::uint8_t> mask_;
  Rational areaSeen_;
  int placed_ = 0;
  int binsUsed_ = 0;
  bool done_ = true;
};

// Baselines (no learning), useful as references.
// First legal action in (slot, rotation, column) order: oldest bin first.
int firstFitAction(const PackingEnv& env);
// Oldest open bin that can take the item; within it, the legal action whose
// settled position has the lowest top edge (then leftmost). Slower.
int greedyLowestAction(const PackingEnv& env);

}  // namespace polypack
