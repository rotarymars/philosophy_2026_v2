#pragma once

// Offline reference packing, for scoring online players.
//
// The offline packer sees every item of the episode in advance and may use as
// many bins as it likes (none is ever closed). It tries many packing
// patterns and keeps the one with the fewest bins. A pattern is an item order
// plus a bin-choice rule; every item is placed bottom-left (each of the 8
// rotations at its lowest, left-most free spot, settled, lowest top edge wins).
//   Orders: largest area first, largest diameter first, largest bounding box
//           first, arrival order, then "noisy area" orders (area times a
//           random factor in [0.7, 1.3], seeded, so results are repeatable).
//   Rules:  first fit (oldest bin that takes the item) or best fit (the
//           fullest bin that takes it).
// Geometry is exact, as everywhere else; randomness only picks the orders.
//
// The true optimum is NP-hard to compute, so this is the best packing found,
// not a proven optimum: optimum <= bins, and lowerBound <= optimum. When
// bins == lowerBound the packing is proven optimal.
//
// Score of an online player on an episode = onlineBins / offline bins
// (1.0 = as good as the offline reference; lower is better).

#include "polypack/bin.hpp"
#include "polypack/env.hpp"
#include "polypack/polygon.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace polypack {

// One piece placed by a pattern: which bin it went into and where it ended up.
struct OfflinePlacement {
  int bin;
  int item;  // index of the piece in arrival order
  Polygon placed;
};

// What one packing pattern did.
struct OfflinePatternRun {
  std::string name;         // e.g. "area/first-fit"
  std::string description;  // the same in plain words
  int bins = 0;
  std::vector<OfflinePlacement> steps;  // in the order the pattern placed them
};

struct OfflineResult {
  int bins = 0;           // bins of the best packing found
  int lowerBound = 0;     // ceil(total item area)
  std::string pattern;    // which pattern produced it, e.g. "area/first-fit"
  int patternsTried = 0;
  std::vector<Bin> packing;
  std::vector<OfflinePatternRun> runs;  // every pattern tried, in pattern order
};

constexpr int kDefaultOfflinePatterns = 16;

// Optional live report of what the offline packer is doing (for the viewer).
// Callbacks run on worker threads, possibly several at once.
struct OfflineObserver {
  // Before anything runs: the patterns planned for an episode.
  std::function<void(int episode, const std::vector<std::string>& names,
                     const std::vector<std::string>& descriptions)> onPlan;
  std::function<void(int episode, int pattern)> onStart;
  std::function<void(int episode, int pattern, const OfflinePlacement& step)> onPlace;
  std::function<void(int episode, int pattern, int bins)> onFinish;
  // A pattern that will not run because an earlier one reached the lower bound.
  std::function<void(int episode, int pattern)> onSkip;
};

// Packs `items` offline with up to `patterns` patterns, using the grid size
// of `config` (settling is always on). Stops early once a packing reaches the
// lower bound, since nothing can beat that.
OfflineResult packOffline(const std::vector<Polygon>& items, const EnvConfig& config,
                          int patterns = kDefaultOfflinePatterns);

// packOffline(episodeItems(config, seed), config, patterns).
OfflineResult packOfflineEpisode(const EnvConfig& config, std::uint64_t seed,
                                 int patterns = kDefaultOfflinePatterns);

// packOffline for many episodes at once, spreading (episode, pattern) work
// over `threads` threads. Same results as calling packOffline on each.
// With stopAtLowerBound = false every pattern is tried even when an earlier
// one already reached the lower bound (the best result is the same; this is
// only for showing the patterns).
std::vector<OfflineResult> packOfflineEpisodes(const std::vector<std::vector<Polygon>>& episodes,
                                               const EnvConfig& config, int patterns, int threads,
                                               bool stopAtLowerBound = true,
                                               const OfflineObserver* observer = nullptr);
// Episodes with seeds seedBase, seedBase + 1, ..., seedBase + count - 1.
std::vector<OfflineResult> packOfflineEpisodes(const EnvConfig& config, std::uint64_t seedBase, int count,
                                               int patterns, int threads);

}  // namespace polypack
