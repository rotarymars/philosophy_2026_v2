#pragma once

// Policy/value network for the packing environment.
//
// For every (bin slot, rotation) pair, the bin image and the rotated item
// image are stacked into a 2-channel picture and encoded by a shared CNN.
// The policy head collapses the height axis, keeping the width axis, and
// turns it into one logit per entry column. The network is therefore
// translation-aware along x, which is the axis the action chooses along.
// The value head pools over all pairs.

#include <torch/torch.h>

namespace polypack::train {

struct ModelConfig {
  int slots = 3;
  int rotations = 8;
  int columns = 16;
  int rasterSize = 32;
  int slotFeatures = 3;
  int globalFeatures = 3;
  int channels = 64;
};

struct PolicyOutput {
  torch::Tensor logits;  // [N, slots * rotations * columns]
  torch::Tensor value;   // [N]
};

class PolicyNetImpl : public torch::nn::Module {
 public:
  explicit PolicyNetImpl(const ModelConfig& config);

  // binImages [N,S,H,W] uint8, itemImages [N,R,H,W] uint8,
  // slotFeatures [N,S,Fs] float, globalFeatures [N,Fg] float.
  PolicyOutput forward(const torch::Tensor& binImages, const torch::Tensor& itemImages,
                       const torch::Tensor& slotFeatures, const torch::Tensor& globalFeatures);

  const ModelConfig& config() const noexcept { return config_; }

 private:
  ModelConfig config_;
  torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr}, conv4_{nullptr};
  torch::nn::Linear condition_{nullptr};
  torch::nn::Conv1d policy1_{nullptr}, policy2_{nullptr};
  torch::nn::Linear value1_{nullptr}, value2_{nullptr};
};
TORCH_MODULE(PolicyNet);

// Illegal actions (mask == false) get a large negative logit.
torch::Tensor maskLogits(const torch::Tensor& logits, const torch::Tensor& mask);

}  // namespace polypack::train
