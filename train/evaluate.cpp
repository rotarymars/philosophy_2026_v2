// Evaluates a trained checkpoint against the baselines on fixed episodes,
// re-verifies every final packing with the exact overlap test, and can write
// SVG pictures of the bins.
//
//   ./polypack_evaluate --model models/policy.pt --episodes 32 --svg-dir eval_svg

#include "common.hpp"

#include "polypack/env.hpp"
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

  --model PATH           checkpoint (default models/policy.pt)
  --episodes N           number of episodes (default 32)
  --seed-base N          episode i uses seed N + i (default 1000000007, the
                         training-time evaluation set; use another value for
                         unseen episodes)
  --device NAME          auto | cpu | cuda | cuda:N (default auto)
  --threads N            CPU threads (default: all cores)
  --greedy-baseline      also run the bottom-left-fill greedy baseline
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
  ObsBatch obs = ObsBatch::allocate(envs[0], episodes);
  std::vector<int64_t> actions(static_cast<std::size_t>(episodes), -1);

  while (!envs[0].done()) {
    if (player == Player::Policy) {
      observeAll(envs, obs, threads);
      torch::NoGradGuard noGrad;
      const ObsBatch d = obs.to(device);
      const torch::Tensor logits =
          maskLogits(runModel(*model, d).logits, d.mask);
      const torch::Tensor chosen =
          (sample ? torch::multinomial(torch::softmax(logits, 1), 1).squeeze(1) : logits.argmax(1))
              .to(torch::kCPU);
      const auto acc = chosen.accessor<int64_t, 1>();
      for (int i = 0; i < episodes; ++i) actions[static_cast<std::size_t>(i)] = acc[i];
    }
    parallelFor(episodes, threads, [&](int i) {
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
    const std::string svgDir = args.str("svg-dir", "");
    const int svgEpisodes = static_cast<int>(args.integer("svg-episodes", 1));
    args.rejectUnknown();

    const CheckpointMeta meta = loadCheckpointMeta(modelPath);
    PolicyNet model(meta.model);
    model->to(device);
    loadCheckpointWeights(modelPath, model, nullptr, device);
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

    std::cout << "\nepisode  policy  first-fit" << (greedy ? "  greedy" : "") << "  area-lower-bound\n";
    bool allValid = true;
    for (int i = 0; i < episodes; ++i) {
      const auto k = static_cast<std::size_t>(i);
      std::cout << std::setw(7) << i << std::setw(8) << policy[k].binsUsed() << std::setw(11)
                << firstFit[k].binsUsed();
      if (greedy) std::cout << std::setw(8) << greedyRun[k].binsUsed();
      std::cout << std::setw(18) << policy[k].areaLowerBound() << '\n';
      allValid = allValid && verify(policy[k]);
    }
    double bound = 0;
    for (const PackingEnv& e : policy) bound += e.areaLowerBound();
    bound /= episodes;
    std::cout << std::fixed << std::setprecision(3) << "\nmean    " << meanBins(policy) << "  "
              << meanBins(firstFit);
    if (greedy) std::cout << "  " << meanBins(greedyRun);
    std::cout << "  " << bound << '\n';
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
