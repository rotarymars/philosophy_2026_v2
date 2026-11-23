// PPO training for online 2D polygon bin packing.
//
//   ./polypack_train --iterations 2000            # train from scratch (GPU if available)
//   ./polypack_train --resume models/policy_latest.pt --iterations 500
//   ./polypack_train --help
//
// Geometry stays exact: the network only *chooses* (bin, rotation, column).
// Legality and the final resting position are computed with exact rationals.

#include "common.hpp"

#include "polypack/env.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

using namespace polypack;
using namespace polypack::train;

namespace {

constexpr uint64_t kEvalSeedBase = 1'000'000'007ULL;

const char* kUsage = R"(polypack_train: PPO training for online 2D polygon bin packing

Output
  --out PATH              best checkpoint (default models/policy.pt); also writes
                          <out>_latest.pt and <out>_log.csv next to it
  --resume PATH           continue from a checkpoint (environment and model
                          settings are taken from it)

Run
  --device NAME           auto | cpu | cuda | cuda:N        (default auto)
  --iterations N          PPO iterations to run             (default 1000)
  --envs N                parallel environments             (default 32)
  --rollout N             steps per environment per iter    (default 64)
  --threads N             CPU threads for the environments  (default: all cores)
  --seed N                                                  (default 1)
  --eval-every N          evaluate/save every N iterations  (default 10)
  --eval-episodes N       fixed evaluation episodes         (default 32)

PPO
  --lr X (3e-4)  --gamma X (1.0)  --lambda X (0.95)  --clip X (0.2)
  --epochs N (4)  --minibatch N (256)  --value-coef X (0.5)
  --entropy-coef X (0.01)  --max-grad-norm X (0.5)

Environment (new runs only)
  --open-bins N (2)  --columns N (16)  --raster N (32)  --items N (60)
  --min-vertices N (3)  --max-vertices N (8)
  --min-radius X (0.08)  --max-radius X (0.22)  --inner-ratio X (0.45)

Model (new runs only)
  --channels N (64)
)";

struct EvalResult {
  double meanBins = 0;
  double meanLowerBound = 0;
};

// Plays the same fixed-seed episodes every time. model == nullptr plays the
// first-fit baseline instead.
EvalResult evaluate(PolicyNet* model, const EnvConfig& config, int episodes,
                    const torch::Device& device, int threads) {
  std::vector<PackingEnv> envs(static_cast<std::size_t>(episodes), PackingEnv(config));
  parallelFor(episodes, threads, [&](int i) { envs[static_cast<std::size_t>(i)].reset(kEvalSeedBase + i); });
  ObsBatch obs = ObsBatch::allocate(envs[0], episodes);
  std::vector<int64_t> actions(static_cast<std::size_t>(episodes), -1);

  while (!envs[0].done()) {  // every episode has the same length
    if (model != nullptr) {
      observeAll(envs, obs, threads);
      torch::NoGradGuard noGrad;
      const ObsBatch d = obs.to(device);
      const PolicyOutput out = (*model)->forward(d.bins, d.items, d.slotF, d.globalF);
      const torch::Tensor best = maskLogits(out.logits, d.mask).argmax(1).to(torch::kCPU);
      const auto acc = best.accessor<int64_t, 1>();
      for (int i = 0; i < episodes; ++i) actions[static_cast<std::size_t>(i)] = acc[i];
    }
    parallelFor(episodes, threads, [&](int i) {
      PackingEnv& env = envs[static_cast<std::size_t>(i)];
      const int action = model != nullptr ? static_cast<int>(actions[static_cast<std::size_t>(i)])
                                          : firstFitAction(env);
      env.step(action);
    });
  }

  EvalResult result;
  for (const PackingEnv& env : envs) {
    result.meanBins += env.binsUsed();
    result.meanLowerBound += env.areaLowerBound();
  }
  result.meanBins /= episodes;
  result.meanLowerBound /= episodes;
  return result;
}

std::string siblingPath(const std::string& out, const std::string& suffix) {
  std::filesystem::path p(out);
  const std::string stem = p.stem().string();
  return (p.parent_path() / (stem + suffix)).string();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args(argc, argv);
    if (args.has("help")) {
      std::cout << kUsage;
      return 0;
    }

    const std::string outPath = args.str("out", "models/policy.pt");
    const std::string latestPath = siblingPath(outPath, "_latest.pt");
    const std::string logPath = siblingPath(outPath, "_log.csv");
    const std::string resumePath = args.str("resume", "");

    const torch::Device device = pickDevice(args.str("device", "auto"));
    const int iterations = static_cast<int>(args.integer("iterations", 1000));
    const int numEnvs = static_cast<int>(args.integer("envs", 32));
    const int rollout = static_cast<int>(args.integer("rollout", 64));
    const int threads = static_cast<int>(args.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
    const uint64_t seed = static_cast<uint64_t>(args.integer("seed", 1));
    const int evalEvery = static_cast<int>(args.integer("eval-every", 10));
    const int evalEpisodes = static_cast<int>(args.integer("eval-episodes", 32));

    const double lr = args.real("lr", 3e-4);
    const double gamma = args.real("gamma", 1.0);
    const double lambda = args.real("lambda", 0.95);
    const double clip = args.real("clip", 0.2);
    const int epochs = static_cast<int>(args.integer("epochs", 4));
    const int minibatch = static_cast<int>(args.integer("minibatch", 256));
    const double valueCoef = args.real("value-coef", 0.5);
    const double entropyCoef = args.real("entropy-coef", 0.01);
    const double maxGradNorm = args.real("max-grad-norm", 0.5);

    EnvConfig envConfig;
    envConfig.openBins = static_cast<int>(args.integer("open-bins", envConfig.openBins));
    envConfig.columns = static_cast<int>(args.integer("columns", envConfig.columns));
    envConfig.rasterSize = static_cast<int>(args.integer("raster", envConfig.rasterSize));
    envConfig.itemsPerEpisode = static_cast<int>(args.integer("items", envConfig.itemsPerEpisode));
    GeneratorConfig& gen = envConfig.generator;
    gen.minVertices = static_cast<int>(args.integer("min-vertices", gen.minVertices));
    gen.maxVertices = static_cast<int>(args.integer("max-vertices", gen.maxVertices));
    gen.minRadius = args.real("min-radius", gen.minRadius);
    gen.maxRadius = args.real("max-radius", gen.maxRadius);
    gen.innerRatio = args.real("inner-ratio", gen.innerRatio);
    const int channels = static_cast<int>(args.integer("channels", 64));
    args.rejectUnknown();

    CheckpointMeta meta;
    if (!resumePath.empty()) {
      meta = loadCheckpointMeta(resumePath);
      envConfig = meta.env;
      std::cout << "Resuming from " << resumePath << " at iteration " << meta.iteration
                << " (environment/model settings come from the checkpoint)\n";
    }
    meta.env = envConfig;
    const PackingEnv prototype(envConfig);
    if (resumePath.empty()) meta.model = modelConfigFor(prototype, channels);

    torch::manual_seed(seed);
    PolicyNet model(meta.model);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr).eps(1e-5));
    if (!resumePath.empty()) loadCheckpointWeights(resumePath, model, &optimizer, device);

    std::size_t parameterCount = 0;
    for (const torch::Tensor& p : model->parameters()) parameterCount += static_cast<std::size_t>(p.numel());
    std::cout << "Device " << device << ", " << parameterCount << " parameters, "
              << prototype.actionCount() << " actions (" << prototype.slots() << " slots x "
              << prototype.rotations() << " rotations x " << prototype.columns() << " columns)\n";

    const EvalResult baseline = evaluate(nullptr, envConfig, evalEpisodes, device, threads);
    std::cout << std::fixed << std::setprecision(3) << "First-fit baseline on " << evalEpisodes
              << " eval episodes: " << baseline.meanBins << " bins (area lower bound "
              << baseline.meanLowerBound << ")\n";

    const bool appendLog = !resumePath.empty() && std::filesystem::exists(logPath);
    if (std::filesystem::path(logPath).has_parent_path()) {
      std::filesystem::create_directories(std::filesystem::path(logPath).parent_path());
    }
    std::ofstream log(logPath, appendLog ? std::ios::app : std::ios::trunc);
    if (!appendLog) {
      log << "iteration,env_steps,episodes,mean_bins,mean_lower_bound,policy_loss,value_loss,"
             "entropy,approx_kl,eval_bins,baseline_bins,seconds\n";
    }

    std::vector<PackingEnv> envs(static_cast<std::size_t>(numEnvs), prototype);
    std::mt19937_64 seeder(seed);
    for (PackingEnv& env : envs) env.reset(seeder());

    const auto startTime = std::chrono::steady_clock::now();
    const int firstIteration = static_cast<int>(meta.iteration) + 1;
    for (int iter = firstIteration; iter < firstIteration + iterations; ++iter) {
      const auto iterStart = std::chrono::steady_clock::now();

      // ---- Collect a rollout -------------------------------------------------
      std::vector<ObsBatch> observations;
      observations.reserve(static_cast<std::size_t>(rollout));
      torch::Tensor actions = torch::zeros({rollout, numEnvs}, torch::kInt64);
      torch::Tensor oldLogp = torch::zeros({rollout, numEnvs});
      torch::Tensor values = torch::zeros({rollout, numEnvs});
      torch::Tensor rewards = torch::zeros({rollout, numEnvs});
      torch::Tensor dones = torch::zeros({rollout, numEnvs});
      std::vector<double> finishedBins;
      std::vector<double> finishedBounds;
      std::mutex statsMutex;

      for (int t = 0; t < rollout; ++t) {
        ObsBatch obs = ObsBatch::allocate(prototype, numEnvs);
        observeAll(envs, obs, threads);
        {
          torch::NoGradGuard noGrad;
          const ObsBatch d = obs.to(device);
          const PolicyOutput out = model->forward(d.bins, d.items, d.slotF, d.globalF);
          const torch::Tensor logp = torch::log_softmax(maskLogits(out.logits, d.mask), 1);
          const torch::Tensor a = torch::multinomial(logp.exp(), 1).squeeze(1);
          actions[t].copy_(a.to(torch::kCPU));
          oldLogp[t].copy_(logp.gather(1, a.unsqueeze(1)).squeeze(1).to(torch::kCPU));
          values[t].copy_(out.value.to(torch::kCPU));
        }
        observations.push_back(std::move(obs));

        std::vector<uint64_t> nextSeeds(static_cast<std::size_t>(numEnvs));
        for (uint64_t& s : nextSeeds) s = seeder();
        // Rows are views, so writing through the accessors fills the rollout tensors.
        const torch::Tensor actRow = actions[t];
        torch::Tensor rewRow = rewards[t];
        torch::Tensor doneRow = dones[t];
        const auto act = actRow.accessor<int64_t, 1>();
        auto rew = rewRow.accessor<float, 1>();
        auto done = doneRow.accessor<float, 1>();
        parallelFor(numEnvs, threads, [&](int e) {
          PackingEnv& env = envs[static_cast<std::size_t>(e)];
          const StepResult r = env.step(static_cast<int>(act[e]));
          rew[e] = r.reward;
          done[e] = r.done ? 1.0f : 0.0f;
          if (r.done) {
            {
              const std::lock_guard<std::mutex> lock(statsMutex);
              finishedBins.push_back(env.binsUsed());
              finishedBounds.push_back(env.areaLowerBound());
            }
            env.reset(nextSeeds[static_cast<std::size_t>(e)]);
          }
        });
      }

      torch::Tensor lastValue;
      {
        ObsBatch obs = ObsBatch::allocate(prototype, numEnvs);
        observeAll(envs, obs, threads);
        torch::NoGradGuard noGrad;
        const ObsBatch d = obs.to(device);
        lastValue = model->forward(d.bins, d.items, d.slotF, d.globalF).value.to(torch::kCPU);
      }
      const auto envEnd = std::chrono::steady_clock::now();

      // ---- Generalised advantage estimation -------------------------------------
      torch::Tensor advantages = torch::zeros({rollout, numEnvs});
      torch::Tensor running = torch::zeros({numEnvs});
      for (int t = rollout - 1; t >= 0; --t) {
        const torch::Tensor nextValue = t == rollout - 1 ? lastValue : values[t + 1];
        const torch::Tensor notDone = 1.0 - dones[t];
        const torch::Tensor delta = rewards[t] + gamma * nextValue * notDone - values[t];
        running = delta + gamma * lambda * notDone * running;
        advantages[t].copy_(running);
      }
      const torch::Tensor returns = advantages + values;

      // ---- PPO update --------------------------------------------------------------
      const int64_t total = static_cast<int64_t>(rollout) * numEnvs;
      const ObsBatch batch = ObsBatch::stack(observations).to(device);
      const torch::Tensor bActions = actions.reshape({total}).to(device);
      const torch::Tensor bOldLogp = oldLogp.reshape({total}).to(device);
      const torch::Tensor bReturns = returns.reshape({total}).to(device);
      torch::Tensor bAdv = advantages.reshape({total}).to(device);
      bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8);

      double policyLossSum = 0, valueLossSum = 0, entropySum = 0, klSum = 0;
      int updates = 0;
      for (int epoch = 0; epoch < epochs; ++epoch) {
        const torch::Tensor perm = torch::randperm(total, torch::TensorOptions().dtype(torch::kInt64).device(device));
        for (int64_t start = 0; start < total; start += minibatch) {
          const torch::Tensor idx = perm.slice(0, start, std::min(start + minibatch, total));
          const ObsBatch mb = batch.index(idx);
          const PolicyOutput out = model->forward(mb.bins, mb.items, mb.slotF, mb.globalF);
          const torch::Tensor logp = torch::log_softmax(maskLogits(out.logits, mb.mask), 1);
          const torch::Tensor newLogp = logp.gather(1, bActions.index_select(0, idx).unsqueeze(1)).squeeze(1);
          const torch::Tensor entropy = -(logp.exp() * logp).sum(1).mean();
          const torch::Tensor oldLp = bOldLogp.index_select(0, idx);
          const torch::Tensor adv = bAdv.index_select(0, idx);
          const torch::Tensor ratio = (newLogp - oldLp).exp();
          const torch::Tensor policyLoss =
              -torch::min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean();
          const torch::Tensor valueLoss = 0.5 * (out.value - bReturns.index_select(0, idx)).pow(2).mean();
          const torch::Tensor loss = policyLoss + valueCoef * valueLoss - entropyCoef * entropy;

          optimizer.zero_grad();
          loss.backward();
          torch::nn::utils::clip_grad_norm_(model->parameters(), maxGradNorm);
          optimizer.step();

          policyLossSum += policyLoss.item<double>();
          valueLossSum += valueLoss.item<double>();
          entropySum += entropy.item<double>();
          klSum += (oldLp - newLogp).mean().item<double>();
          ++updates;
        }
      }
      meta.iteration = iter;
      meta.envSteps += total;

      // ---- Report, evaluate, save ----------------------------------------------------
      const auto iterEnd = std::chrono::steady_clock::now();
      const double envSeconds = std::chrono::duration<double>(envEnd - iterStart).count();
      const double learnSeconds = std::chrono::duration<double>(iterEnd - envEnd).count();
      const double elapsed = std::chrono::duration<double>(iterEnd - startTime).count();
      double meanBins = 0, meanBound = 0;
      for (std::size_t i = 0; i < finishedBins.size(); ++i) {
        meanBins += finishedBins[i];
        meanBound += finishedBounds[i];
      }
      if (!finishedBins.empty()) {
        meanBins /= static_cast<double>(finishedBins.size());
        meanBound /= static_cast<double>(finishedBins.size());
      }

      std::string evalText = "";
      double evalBins = -1;
      if (iter % evalEvery == 0 || iter == firstIteration + iterations - 1) {
        evalBins = evaluate(&model, envConfig, evalEpisodes, device, threads).meanBins;
        const bool improved = evalBins < meta.bestEvalBins;
        if (improved) {
          meta.bestEvalBins = evalBins;
          saveCheckpoint(outPath, model, &optimizer, meta);
        }
        saveCheckpoint(latestPath, model, &optimizer, meta);
        std::ostringstream s;
        s << std::fixed << std::setprecision(3) << " | eval " << evalBins << " bins (first-fit "
          << baseline.meanBins << ")" << (improved ? " * saved best" : "");
        evalText = s.str();
      }

      std::cout << std::fixed << std::setprecision(3) << "iter " << iter << " | steps " << meta.envSteps
                << " | episodes " << finishedBins.size() << " bins " << meanBins << " (lb " << meanBound
                << ") | pi " << policyLossSum / updates << " v " << valueLossSum / updates << " H "
                << entropySum / updates << " kl " << klSum / updates << " | env " << envSeconds
                << "s learn " << learnSeconds << "s" << evalText << std::endl;
      log << iter << ',' << meta.envSteps << ',' << finishedBins.size() << ',' << meanBins << ','
          << meanBound << ',' << policyLossSum / updates << ',' << valueLossSum / updates << ','
          << entropySum / updates << ',' << klSum / updates << ',' << evalBins << ','
          << baseline.meanBins << ',' << elapsed << '\n';
      log.flush();
    }

    std::cout << "Done. Best eval " << meta.bestEvalBins << " bins; best model: " << outPath
              << ", latest: " << latestPath << ", log: " << logPath << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
