// Plays one episode and streams every step as a JSON line on stdout, for the
// live viewer in viz/ (run `python3 viz/server.py`).
//
//   ./polypack_visualize --player policy --model models/policy.pt --seed 7
//   ./polypack_visualize --player greedy --seed 7
//
// Geometry stays exact: the settling path is rebuilt with the same exact
// maxSlide calls that settleBottomLeft uses, and checked against
// env.simulate(). Coordinates are converted to double only when printed.

#include "common.hpp"

#include "polypack/env.hpp"
#include "polypack/slide.hpp"

#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace polypack;
using namespace polypack::train;

namespace {

const char* kUsage = R"(polypack_visualize: stream one episode as JSON lines

  --player NAME     policy | firstfit | greedy (default policy)
  --model PATH      checkpoint for --player policy (default models/policy.pt)
  --seed N          episode seed (default 1)
  --sample          sample from the policy instead of taking the argmax
  --device NAME     auto | cpu | cuda (default cpu)
  Baselines only (a policy uses the settings it was trained with):
  --items N         items per episode (default 60)
  --open-bins N     open bins (default 2)
  --grid N          placement grid size (default 16)
  --settle 0|1      slide down/left after placing (default 1)
)";

void writePoly(std::ostream& o, const Polygon& p) {
  o << '[';
  bool first = true;
  for (const Point& v : p.vertices()) {
    if (!first) o << ',';
    first = false;
    o << '[' << toDouble(v.x) << ',' << toDouble(v.y) << ']';
  }
  o << ']';
}

// Mirrors settleBottomLeft, recording each resting point.
std::vector<Polygon> settlePath(Polygon p, const std::vector<Polygon>& obstacles, bool settle) {
  std::vector<Polygon> path{p};
  if (!settle) return path;
  const Box unit = unitSquare();
  for (int round = 0; round < 32; ++round) {
    bool moved = false;
    for (Direction d : {Direction::Down, Direction::Left}) {
      const Rational s = maxSlide(p, d, obstacles, unit);
      if (s > 0) {
        p = p.translated(d == Direction::Down ? Point(Rational(0), Rational(-s)) : Point(Rational(-s), Rational(0)));
        path.push_back(p);
        moved = true;
      }
    }
    if (!moved) break;
  }
  return path;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args(argc, argv);
    if (args.has("help")) {
      std::cout << kUsage;
      return 0;
    }
    const std::string player = args.str("player", "policy");
    const std::string modelPath = args.str("model", "models/policy.pt");
    const uint64_t seed = static_cast<uint64_t>(args.integer("seed", 1));
    const bool sample = args.has("sample");
    args.str("sample", "");
    const torch::Device device = pickDevice(args.str("device", "cpu"));
    EnvConfig config;
    config.itemsPerEpisode = static_cast<int>(args.integer("items", config.itemsPerEpisode));
    config.openBins = static_cast<int>(args.integer("open-bins", config.openBins));
    config.grid = static_cast<int>(args.integer("grid", config.grid));
    config.settle = args.integer("settle", 1) != 0;
    args.rejectUnknown();
    if (player != "policy" && player != "firstfit" && player != "greedy") {
      throw std::invalid_argument("unknown --player " + player);
    }

    std::optional<PolicyNet> model;
    if (player == "policy") {
      const CheckpointMeta meta = loadCheckpointMeta(modelPath);
      config = meta.env;
      model.emplace(meta.model);
      (*model)->to(device);
      loadCheckpointWeights(modelPath, *model, nullptr, device);
      (*model)->eval();  // torch module inference mode (not code evaluation)
    }
    // A single episode: use every core for the legality mask.
    config.maskThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    PackingEnv env(config);
    env.reset(seed);
    const int G = env.grid();
    std::cout.precision(9);
    std::cout << "{\"type\":\"start\",\"player\":\"" << player << "\",\"seed\":" << seed
              << ",\"openBins\":" << config.openBins << ",\"grid\":" << G
              << ",\"settle\":" << (config.settle ? "true" : "false")
              << ",\"rotations\":" << env.rotations() << ",\"items\":" << config.itemsPerEpisode << "}\n"
              << std::flush;

    ObsBatch obs = ObsBatch::allocate(env, 1);
    while (!env.done()) {
      int action = 0;
      torch::Tensor probs;
      if (player == "policy") {
        observeInto(env, obs, 0);
        torch::NoGradGuard noGrad;
        const ObsBatch d = obs.to(device);
        const torch::Tensor logits = maskLogits(runModel(*model, d).logits, d.mask);
        probs = torch::softmax(logits, 1).squeeze(0).to(torch::kCPU);
        action = static_cast<int>(
            (sample ? torch::multinomial(probs, 1).item<int64_t>() : probs.argmax().item<int64_t>()));
      } else if (player == "firstfit") {
        action = firstFitAction(env);
      } else {
        action = greedyBottomLeftAction(env);
      }

      const Action act = env.decodeAction(action);
      const bool newBin = act.slot == config.openBins;
      const int binsBefore = env.binsUsed();
      const int openCount = std::min(binsBefore, config.openBins);
      const int firstOpen = binsBefore - openCount;  // open bins are always the newest ones
      const int targetBin = newBin ? binsBefore : firstOpen + act.slot;
      const int openFrom = newBin && openCount == config.openBins ? firstOpen + 1 : firstOpen;

      const std::vector<Polygon> path = settlePath(env.anchorPlacement(action), env.slotItems(act.slot), config.settle);
      if (!(path.back() == env.simulate(action))) {
        throw std::logic_error("settle path disagrees with env.simulate");
      }

      // Where this item could have gone, for the chosen bin and rotation.
      const std::vector<std::uint8_t>& mask = env.mask();
      const int base = env.encodeAction(act.slot, act.rotation, 0, 0);
      std::string legalMap(static_cast<std::size_t>(G * G), '0');
      for (int k = 0; k < G * G; ++k) {
        if (mask[static_cast<std::size_t>(base + k)]) legalMap[static_cast<std::size_t>(k)] = '1';
      }
      int legalCount = 0;
      for (std::uint8_t v : mask) legalCount += v;

      std::ostringstream o;
      o.precision(6);
      o << "{\"type\":\"step\",\"index\":" << env.itemsPlaced() << ",\"slot\":" << act.slot
        << ",\"newBin\":" << (newBin ? "true" : "false") << ",\"rotation\":" << act.rotation
        << ",\"gx\":" << act.gx << ",\"gy\":" << act.gy << ",\"targetBin\":" << targetBin
        << ",\"openFrom\":" << openFrom << ",\"legalCount\":" << legalCount
        << ",\"legalMap\":\"" << legalMap << "\",\"path\":[";
      o.precision(9);
      for (std::size_t i = 0; i < path.size(); ++i) {
        if (i) o << ',';
        writePoly(o, path[i]);
      }
      o << ']';
      o.precision(6);
      if (probs.defined()) {
        const auto pa = probs.accessor<float, 1>();
        o << ",\"probMap\":[";
        for (int k = 0; k < G * G; ++k) o << (k ? "," : "") << pa[base + k];
        o << "],\"slotProbs\":[";
        const int perSlot = env.rotations() * env.positions();
        for (int s = 0; s < env.slots(); ++s) {
          double sum = 0;
          for (int k = 0; k < perSlot; ++k) sum += pa[s * perSlot + k];
          o << (s ? "," : "") << sum;
        }
        o << "],\"chosenProb\":" << pa[action];
      }
      env.step(action);
      const std::vector<Bin> after = env.allBins();
      o << ",\"binsUsed\":" << env.binsUsed() << ",\"lowerBound\":" << env.areaLowerBound()
        << ",\"fill\":" << toDouble(after[static_cast<std::size_t>(targetBin)].fillRatio()) << "}\n";
      std::cout << o.str() << std::flush;
    }
    std::cout << "{\"type\":\"done\",\"binsUsed\":" << env.binsUsed()
              << ",\"lowerBound\":" << env.areaLowerBound() << "}\n" << std::flush;
    return 0;
  } catch (const std::exception& e) {
    std::cout << "{\"type\":\"error\",\"message\":\"" << e.what() << "\"}\n" << std::flush;
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
