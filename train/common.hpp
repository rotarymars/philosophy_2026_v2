#pragma once

// Shared helpers for the training and evaluation tools.

#include "model.hpp"

#include "polypack/env.hpp"
#include "polypack/offline.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace polypack::train {

// ---- Checkpoints --------------------------------------------------------------
//
// A checkpoint is self-describing: it stores the environment and model
// configuration next to the weights, so evaluation recreates exactly the
// setting the model was trained in.

struct CheckpointMeta {
  EnvConfig env;
  ModelConfig model;
  int64_t iteration = 0;
  int64_t envSteps = 0;
  double bestEvalScore = 1e30;  // best mean (bins / offline bins) so far
};

ModelConfig modelConfigFor(const PackingEnv& env, int channels);

void saveCheckpoint(const std::string& path, PolicyNet& model, torch::optim::Optimizer* optimizer,
                    const CheckpointMeta& meta);

// Reads only the metadata (to build the model before loading weights).
CheckpointMeta loadCheckpointMeta(const std::string& path);

// Loads weights (and optimizer state, if given and present) into `model`,
// which must already live on `device`.
void loadCheckpointWeights(const std::string& path, PolicyNet& model,
                           torch::optim::Optimizer* optimizer, const torch::Device& device);

// ---- Observations -------------------------------------------------------------

struct ObsBatch {
  torch::Tensor bins;     // [N,S,H,W] uint8
  torch::Tensor items;    // [N,R,H,W] uint8
  torch::Tensor contact;  // [N,S*R,H,W] uint8
  torch::Tensor slotF;    // [N,S,Fs] float
  torch::Tensor globalF;  // [N,Fg] float
  torch::Tensor mask;     // [N,A] bool
  torch::Tensor critic;   // [N,Fc] float, critic-only features (never shown to the policy)

  static ObsBatch allocate(const PackingEnv& like, int64_t n);
  ObsBatch to(const torch::Device& device) const;
  static ObsBatch stack(const std::vector<ObsBatch>& steps);  // [T] x [N,...] -> [T*N,...]
  ObsBatch index(const torch::Tensor& indices) const;
};

// Offline reference results for the episodes with seeds seedBase ..
// seedBase + count - 1, read from the cache file at `cachePath` when they are
// there and computed (then appended to the file) when not. The offline
// reference is deterministic, so a cached result is exactly what computing
// it again would give; the cache key holds every setting it depends on.
// Cached results carry bins, lowerBound and pattern (no packing).
// An empty cachePath disables the cache.
constexpr const char* kDefaultOfflineCache = "models/offline_cache.tsv";
std::vector<OfflineResult> offlineReference(const EnvConfig& config, std::uint64_t seedBase, int count, int patterns,
                                            int threads, const std::string& cachePath);

// Fills row i of `out` from one environment.
void observeInto(const PackingEnv& env, ObsBatch& out, int64_t i);

// Fills `out` (allocated for envs.size()) from the environments, in parallel.
void observeAll(const std::vector<PackingEnv>& envs, ObsBatch& out, int threads);
// Indices of the episodes that are not finished yet (episodes differ in length).
std::vector<int> unfinished(const std::vector<PackingEnv>& envs);
// Observations of envs[indices[k]] in row k.
ObsBatch observeSome(const std::vector<PackingEnv>& envs, const std::vector<int>& indices, int threads);

// Runs the model on a batch that already lives on the model's device.
PolicyOutput runModel(PolicyNet& model, const ObsBatch& batch);

// ---- Misc ---------------------------------------------------------------------

// Runs fn(i) for i in [0, n) on up to `threads` threads.
void parallelFor(int n, int threads, const std::function<void(int)>& fn);

torch::Device pickDevice(const std::string& name);  // "auto", "cpu", "cuda", "cuda:1", ...

// Minimal "--key value" command-line parser.
class Args {
 public:
  Args(int argc, char** argv);
  std::string str(const std::string& key, const std::string& fallback) const;
  int64_t integer(const std::string& key, int64_t fallback) const;
  double real(const std::string& key, double fallback) const;
  bool has(const std::string& key) const { return values_.count(key) != 0; }
  // Throws if an option was given that no one asked for (catches typos).
  void rejectUnknown() const;

 private:
  std::map<std::string, std::string> values_;
  mutable std::map<std::string, bool> used_;
};

}  // namespace polypack::train
