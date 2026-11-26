// Evaluates a trained checkpoint against the baselines on fixed episodes,
// re-verifies every final packing with the exact overlap test, and can write
// SVG pictures of the bins.
//
// Score of a player on an episode = bins it used / bins of the offline
// reference packing (which sees all items first and tries many packing
// patterns; see polypack/offline.hpp). The reported score is the mean over
// episodes; lower is better, 1.0 means as good as the offline reference.
//
//   ./polypack_evaluate --model models/policy.pt --episodes 32 --svg-dir eval_svg

#include "common.hpp"

#include "polypack/env.hpp"
#include "polypack/offline.hpp"
#include "polypack/overlap.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace polypack;
using namespace polypack::train;

namespace {

const char* kUsage = R"(polypack_evaluate: compare a trained policy with baselines

Score = mean over episodes of (bins used / bins of the offline reference
packing, which sees every item first and tries many packing patterns).
Lower is better; 1.0 = as good as the offline reference.

  --model PATH           checkpoint (default models/policy.pt)
  --episodes N           number of episodes (default 32)
  --seed-base N          episode i uses seed N + i (default 1000000007, the
                         training-time evaluation set; use another value for
                         unseen episodes)
  --device NAME          auto | cpu | cuda | cuda:N (default auto)
  --threads N            CPU threads (default: all cores)
  --greedy-baseline      also run the bottom-left-fill greedy baseline
  --offline-patterns N   packing patterns the offline reference tries per
                         episode (default 16)
  --min-items N          fewest items per episode (default 20)
  --items N              most items per episode (default 100); each episode
                         gets a random count in between, chosen by its seed.
                         The same for every model, whatever it was trained on
  --sample               sample from the policy instead of taking the argmax
  --svg-dir DIR          write SVGs of the policy's bins for the first episodes
  --svg-episodes N       how many episodes to draw (default 1)
)";

enum class Player { Policy, FirstFit, Greedy };

std::vector<PackingEnv> play(Player player, PolicyNet* model, const EnvConfig& config,
                             uint64_t seedBase, int episodes, bool sample,
                             const torch::Device& device, int threads) {
  std::vector<PackingEnv> envs(static_cast<std::size_t>(episodes), PackingEnv(config));
  parallelFor(episodes, threads, [&](int i) { envs[static_cast<std::size_t>(i)].reset(seedBase + i); });
  std::vector<int64_t> actions(static_cast<std::size_t>(episodes), -1);

  // Episodes differ in length: each step only plays the unfinished ones.
  for (std::vector<int> active = unfinished(envs); !active.empty(); active = unfinished(envs)) {
    const int n = static_cast<int>(active.size());
    if (player == Player::Policy) {
      const ObsBatch obs = observeSome(envs, active, threads);
      torch::NoGradGuard noGrad;
      const ObsBatch d = obs.to(device);
      const torch::Tensor logits =
          maskLogits(runModel(*model, d).logits, d.mask);
      const torch::Tensor chosen =
          (sample ? torch::multinomial(torch::softmax(logits, 1), 1).squeeze(1) : logits.argmax(1))
              .to(torch::kCPU);
      const auto acc = chosen.accessor<int64_t, 1>();
      for (int k = 0; k < n; ++k) actions[static_cast<std::size_t>(active[static_cast<std::size_t>(k)])] = acc[k];
    }
    parallelFor(n, threads, [&](int k) {
      const int i = active[static_cast<std::size_t>(k)];
      PackingEnv& env = envs[static_cast<std::size_t>(i)];
      int action = 0;
      switch (player) {
        case Player::Policy: action = static_cast<int>(actions[static_cast<std::size_t>(i)]); break;
        case Player::FirstFit: action = firstFitAction(env); break;
        case Player::Greedy: action = greedyBottomLeftAction(env); break;
      }
      env.step(action);
    });
  }
  return envs;
}

// Independent exact re-check of a finished episode.
bool verify(const PackingEnv& env) {
  for (const Bin& bin : env.allBins()) {
    const std::vector<Polygon>& items = bin.items();
    for (std::size_t i = 0; i < items.size(); ++i) {
      if (!isContainedIn(items[i], bin.bounds())) return false;
      for (std::size_t j = i + 1; j < items.size(); ++j) {
        if (interiorsOverlap(items[i], items[j])) return false;
      }
    }
  }
  return true;
}

double meanBins(const std::vector<PackingEnv>& envs) {
  double sum = 0;
  for (const PackingEnv& e : envs) sum += e.binsUsed();
  return sum / static_cast<double>(envs.size());
}

double meanScore(const std::vector<PackingEnv>& envs, const std::vector<OfflineResult>& offline) {
  double sum = 0;
  for (std::size_t i = 0; i < envs.size(); ++i) {
    sum += static_cast<double>(envs[i].binsUsed()) / offline[i].bins;
  }
  return sum / static_cast<double>(envs.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args(argc, argv);
    if (args.has("help")) {
      std::cout << kUsage;
      return 0;
    }
    const std::string modelPath = args.str("model", "models/policy.pt");
    const int episodes = static_cast<int>(args.integer("episodes", 32));
    const uint64_t seedBase = static_cast<uint64_t>(args.integer("seed-base", 1'000'000'007LL));
    const torch::Device device = pickDevice(args.str("device", "auto"));
    const int threads = static_cast<int>(args.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
    const bool greedy = args.has("greedy-baseline");
    args.str("greedy-baseline", "");
    const bool sample = args.has("sample");
    args.str("sample", "");
    const int offlinePatterns = static_cast<int>(args.integer("offline-patterns", kDefaultOfflinePatterns));
    EnvConfig defaults;
    const int maxItems = static_cast<int>(args.integer("items", defaults.itemsPerEpisode));
    const int minItems = static_cast<int>(args.integer("min-items", defaults.minItemsPerEpisode));
    const std::string svgDir = args.str("svg-dir", "");
    const int svgEpisodes = static_cast<int>(args.integer("svg-episodes", 1));
    args.rejectUnknown();

    CheckpointMeta meta = loadCheckpointMeta(modelPath);
    // The network does not depend on the item count, so every model is
    // judged on the same episodes.
    meta.env.itemsPerEpisode = maxItems;
    meta.env.minItemsPerEpisode = minItems;
    PolicyNet model(meta.model);
    model->to(device);
    loadCheckpointWeights(modelPath, model, nullptr, device);
    std::cout << "Episodes: " << episodes << ", each with a random number of pieces from " << minItems
              << " to " << maxItems << " (chosen by its seed)\n";
    std::cout << "Model " << modelPath << " (iteration " << meta.iteration << ", " << meta.envSteps
              << " env steps)\n";

    const std::vector<PackingEnv> policy =
        play(Player::Policy, &model, meta.env, seedBase, episodes, sample, device, threads);
    const std::vector<PackingEnv> firstFit =
        play(Player::FirstFit, nullptr, meta.env, seedBase, episodes, false, device, threads);
    std::vector<PackingEnv> greedyRun;
    if (greedy) {
      greedyRun = play(Player::Greedy, nullptr, meta.env, seedBase, episodes, false, device, threads);
    }

    const std::vector<OfflineResult> offline =
        packOfflineEpisodes(meta.env, seedBase, episodes, offlinePatterns, threads);

    std::cout << "\nbins used per episode\n"
              << "episode  pieces  policy  first-fit" << (greedy ? "  greedy" : "")
              << "  offline  area-lower-bound  offline pattern\n";
    bool allValid = true;
    int proven = 0;
    for (int i = 0; i < episodes; ++i) {
      const auto k = static_cast<std::size_t>(i);
      std::cout << std::setw(7) << i << std::setw(8) << policy[k].episodeItemCount() << std::setw(8)
                << policy[k].binsUsed() << std::setw(11)
                << firstFit[k].binsUsed();
      if (greedy) std::cout << std::setw(8) << greedyRun[k].binsUsed();
      std::cout << std::setw(9) << offline[k].bins << std::setw(18) << offline[k].lowerBound << "  "
                << offline[k].pattern << '\n';
      allValid = allValid && verify(policy[k]);
      if (offline[k].bins == offline[k].lowerBound) ++proven;
    }
    double bound = 0, offlineMean = 0;
    for (const OfflineResult& r : offline) {
      bound += r.lowerBound;
      offlineMean += r.bins;
    }
    bound /= episodes;
    offlineMean /= episodes;
    double meanPieces = 0;
    for (const PackingEnv& e : policy) meanPieces += e.episodeItemCount();
    meanPieces /= episodes;
    std::cout << std::fixed << std::setprecision(3) << "   mean" << std::setw(8) << std::setprecision(1)
              << meanPieces << std::setprecision(3) << std::setw(8) << meanBins(policy)
              << std::setw(11) << meanBins(firstFit);
    if (greedy) std::cout << std::setw(8) << meanBins(greedyRun);
    std::cout << std::setw(9) << offlineMean << std::setw(18) << bound << '\n';

    std::cout << "\nscore = mean(bins used / offline bins), lower is better\n"
              << "  policy     " << meanScore(policy, offline) << '\n'
              << "  first-fit  " << meanScore(firstFit, offline) << '\n';
    if (greedy) std::cout << "  greedy     " << meanScore(greedyRun, offline) << '\n';
    std::cout << "offline reference is proven optimal (= area lower bound) on " << proven << " of "
              << episodes << " episodes\n";
    std::cout << "exact re-check of every policy packing: " << (allValid ? "OK, no overlaps" : "FAILED")
              << '\n';

    if (!svgDir.empty()) {
      std::filesystem::create_directories(svgDir);
      for (int i = 0; i < std::min(svgEpisodes, episodes); ++i) {
        const std::vector<Bin> bins = policy[static_cast<std::size_t>(i)].allBins();
        for (std::size_t b = 0; b < bins.size(); ++b) {
          const std::string path = svgDir + "/episode" + std::to_string(i) + "_bin" + std::to_string(b) + ".svg";
          std::ofstream(path) << bins[b].toSvg();
        }
      }
      std::cout << "SVGs written to " << svgDir << '\n';
    }
    return allValid ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
