#include "polypack/offline.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <thread>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>

namespace polypack {
namespace {

Rational squaredDiameter(const Polygon& p) {
  Rational best = 0;
  const std::vector<Point>& v = p.vertices();
  for (std::size_t i = 0; i < v.size(); ++i) {
    for (std::size_t j = i + 1; j < v.size(); ++j) {
      const Rational dx = v[i].x - v[j].x;
      const Rational dy = v[i].y - v[j].y;
      const Rational d = dx * dx + dy * dy;
      if (d > best) best = d;
    }
  }
  return best;
}

// Indices sorted by key, largest first; ties keep arrival order.
std::vector<std::size_t> descendingBy(const std::vector<Rational>& key) {
  std::vector<std::size_t> idx(key.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return key[a] > key[b]; });
  return idx;
}

enum class Rule { FirstFit, BestFit };

// Bottom-left placement in the bin chosen by `rule` (the new-bin slot only
// when no open bin takes the item).
int chooseAction(const PackingEnv& env, Rule rule) {
  if (rule == Rule::FirstFit) return greedyBottomLeftAction(env);
  const std::vector<std::uint8_t>& mask = env.mask();
  const int openCount = std::min(env.binsUsed(), env.config().openBins);
  const std::vector<Bin> bins = env.allBins();  // no bin is ever closed offline: slot s == bins[s]
  std::vector<int> slots(static_cast<std::size_t>(openCount));
  std::iota(slots.begin(), slots.end(), 0);
  std::stable_sort(slots.begin(), slots.end(), [&](int a, int b) {
    return bins[static_cast<std::size_t>(a)].usedArea() > bins[static_cast<std::size_t>(b)].usedArea();
  });
  slots.push_back(env.config().openBins);
  for (int s : slots) {
    int best = -1;
    std::optional<Polygon> bestPlacement;
    for (int r = 0; r < env.rotations(); ++r) {
      const int base = env.encodeAction(s, r, 0, 0);
      int first = -1;
      for (int k = 0; k < env.positions(); ++k) {
        if (mask[static_cast<std::size_t>(base + k)]) {
          first = base + k;
          break;
        }
      }
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
  throw std::logic_error("offline best fit: no legal action");
}

struct Pattern {
  std::string name;
  std::vector<std::size_t> order;
  Rule rule;
};

// The first `count` patterns for these items (deterministic).
std::vector<Pattern> makePatterns(const std::vector<Polygon>& items, int count) {
  std::vector<Rational> area, diameter, boxArea;
  for (const Polygon& p : items) {
    area.push_back(p.area());
    diameter.push_back(squaredDiameter(p));
    const Box& b = p.bounds();
    boxArea.push_back(Rational((b.max.x - b.min.x) * (b.max.y - b.min.y)));
  }
  std::vector<std::size_t> arrival(items.size());
  std::iota(arrival.begin(), arrival.end(), std::size_t{0});
  std::vector<std::pair<std::string, std::vector<std::size_t>>> orders = {
      {"area", descendingBy(area)},
      {"diameter", descendingBy(diameter)},
      {"bbox", descendingBy(boxArea)},
      {"arrival", arrival}};
  std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);  // fixed: same patterns every time
  while (static_cast<int>(orders.size()) * 2 < count) {
    std::vector<Rational> noisy;
    for (const Rational& a : area) {
      noisy.push_back(Rational(a * Rational(static_cast<long>(70 + rng() % 61), 100)));
    }
    orders.emplace_back("noisy-area-" + std::to_string(orders.size() - 3), descendingBy(noisy));
  }
  std::vector<Pattern> patterns;
  for (const auto& [name, order] : orders) {
    for (Rule rule : {Rule::FirstFit, Rule::BestFit}) {
      if (static_cast<int>(patterns.size()) >= count) return patterns;
      patterns.push_back({name + (rule == Rule::FirstFit ? "/first-fit" : "/best-fit"), order, rule});
    }
  }
  return patterns;
}

int lowerBoundOf(const std::vector<Polygon>& items) {
  Rational total = 0;
  for (const Polygon& p : items) total += p.area();
  mpz_class bound;
  mpz_cdiv_q(bound.get_mpz_t(), total.get_num_mpz_t(), total.get_den_mpz_t());
  return static_cast<int>(bound.get_si());
}

// Packs the items with one pattern; returns the bins.
std::vector<Bin> runPattern(const std::vector<Polygon>& items, const EnvConfig& config, const Pattern& pattern) {
  EnvConfig offline = config;
  offline.openBins = static_cast<int>(items.size());  // never close a bin
  offline.itemsPerEpisode = static_cast<int>(items.size());
  offline.minItemsPerEpisode = 0;
  offline.settle = true;
  offline.maskThreads = 1;
  std::vector<Polygon> sequence;
  sequence.reserve(items.size());
  for (std::size_t i : pattern.order) sequence.push_back(items[i]);
  PackingEnv env(offline);
  env.resetWithItems(std::move(sequence));
  while (!env.done()) env.step(chooseAction(env, pattern.rule));
  return env.allBins();
}

void runTasks(int n, int threads, const std::function<void(int)>& fn) {
  const int workers = std::max(1, std::min(threads, n));
  std::atomic<int> next{0};
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
  std::vector<std::thread> pool;
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      try {
        for (int t = next++; t < n; t = next++) fn(t);
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

}  // namespace

OfflineResult packOffline(const std::vector<Polygon>& items, const EnvConfig& config, int patterns) {
  return packOfflineEpisodes({items}, config, patterns, 1).front();
}

OfflineResult packOfflineEpisode(const EnvConfig& config, std::uint64_t seed, int patterns) {
  return packOffline(episodeItems(config, seed), config, patterns);
}

std::vector<OfflineResult> packOfflineEpisodes(const std::vector<std::vector<Polygon>>& episodes,
                                               const EnvConfig& config, int patterns, int threads) {
  if (patterns < 1) throw std::invalid_argument("packOffline: patterns must be >= 1");
  const int n = static_cast<int>(episodes.size());
  std::vector<OfflineResult> results(episodes.size());
  std::vector<std::vector<Pattern>> pats(episodes.size());
  for (std::size_t e = 0; e < episodes.size(); ++e) {
    pats[e] = makePatterns(episodes[e], patterns);
    results[e].lowerBound = lowerBoundOf(episodes[e]);
  }

  // Round 1: the first pattern for every episode. Round 2: the other
  // patterns, only for episodes not yet at the lower bound, spread over all
  // threads as (episode, pattern) tasks. Keeping the smallest bin count and,
  // among equals, the earliest pattern gives the same answer as trying the
  // patterns one by one and stopping at the lower bound.
  std::vector<std::vector<std::optional<std::vector<Bin>>>> packs(episodes.size());
  for (std::size_t e = 0; e < episodes.size(); ++e) packs[e].resize(pats[e].size());
  runTasks(n, threads, [&](int e) {
    const auto k = static_cast<std::size_t>(e);
    if (!episodes[k].empty()) packs[k][0] = runPattern(episodes[k], config, pats[k][0]);
  });
  std::vector<std::pair<std::size_t, std::size_t>> tasks;
  for (std::size_t e = 0; e < episodes.size(); ++e) {
    if (episodes[e].empty()) continue;
    if (static_cast<int>(packs[e][0]->size()) <= results[e].lowerBound) continue;
    for (std::size_t p = 1; p < pats[e].size(); ++p) tasks.emplace_back(e, p);
  }
  runTasks(static_cast<int>(tasks.size()), threads, [&](int t) {
    const auto [e, p] = tasks[static_cast<std::size_t>(t)];
    packs[e][p] = runPattern(episodes[e], config, pats[e][p]);
  });

  for (std::size_t e = 0; e < episodes.size(); ++e) {
    OfflineResult& r = results[e];
    for (std::size_t p = 0; p < packs[e].size(); ++p) {
      if (!packs[e][p]) continue;
      ++r.patternsTried;
      const int bins = static_cast<int>(packs[e][p]->size());
      if (r.packing.empty() || bins < r.bins) {
        r.bins = bins;
        r.pattern = pats[e][p].name;
        r.packing = *packs[e][p];
      }
    }
  }
  return results;
}

std::vector<OfflineResult> packOfflineEpisodes(const EnvConfig& config, std::uint64_t seedBase, int count,
                                               int patterns, int threads) {
  std::vector<std::vector<Polygon>> episodes;
  for (int i = 0; i < count; ++i) episodes.push_back(episodeItems(config, seedBase + static_cast<std::uint64_t>(i)));
  return packOfflineEpisodes(episodes, config, patterns, threads);
}

}  // namespace polypack
