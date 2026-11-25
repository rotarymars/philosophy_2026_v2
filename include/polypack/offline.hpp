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
#include <string>
#include <vector>

namespace polypack {

struct OfflineResult {
  int bins = 0;           // bins of the best packing found
  int lowerBound = 0;     // ceil(total item area)
  std::string pattern;    // which pattern produced it, e.g. "area/first-fit"
  int patternsTried = 0;
  std::vector<Bin> packing;
};

constexpr int kDefaultOfflinePatterns = 16;

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
std::vector<OfflineResult> packOfflineEpisodes(const std::vector<std::vector<Polygon>>& episodes,
                                               const EnvConfig& config, int patterns, int threads);
// Episodes with seeds seedBase, seedBase + 1, ..., seedBase + count - 1.
std::vector<OfflineResult> packOfflineEpisodes(const EnvConfig& config, std::uint64_t seedBase, int count,
                                               int patterns, int threads);

}  // namespace polypack
