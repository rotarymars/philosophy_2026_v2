#pragma once

// Policy/value network for the packing environment (free placement).
//
// For every (bin slot, rotation) pair the network looks at a stack of
// pictures at raster resolution:
//   * the bin's occupancy,
//   * the contact hint (how snugly the item would sit at each spot),
// and, after one stride-2 convolution aligned with the placement grid, the
// exact legality mask of that pair as an extra channel. A small summary of
// the item's shape and the slot/global features are added to every position.
//
// The policy head is fully convolutional and outputs one logit per grid spot,
// so every spot is scored by looking at its own surroundings. The value head
// pools over all pairs.

#include <torch/torch.h>

namespace polypack::train {

struct ModelConfig {
  int slots = 3;
  int rotations = 8;
  int grid = 16;
  int rasterSize = 32;
  int slotFeatures = 3;
  int globalFeatures = 3;
  int channels = 64;
};

struct PolicyOutput {
  torch::Tensor logits;  // [N, slots * rotations * grid * grid]
  torch::Tensor value;   // [N]
};

class PolicyNetImpl : public torch::nn::Module {
 public:
  explicit PolicyNetImpl(const ModelConfig& config);

  // binImages [N,S,H,W] uint8, itemImages [N,R,H,W] uint8,
  // contactImages [N,S*R,H,W] uint8, slotFeatures [N,S,Fs] float,
  // globalFeatures [N,Fg] float, mask [N, S*R*G*G] bool.
  PolicyOutput forward(const torch::Tensor& binImages, const torch::Tensor& itemImages,
                       const torch::Tensor& contactImages, const torch::Tensor& slotFeatures,
                       const torch::Tensor& globalFeatures, const torch::Tensor& mask);

  const ModelConfig& config() const noexcept { return config_; }

 private:
  static constexpr int kItemSummary = 8;  // item shape is summarised as an 8x8 thumbnail

  ModelConfig config_;
  torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr}, conv4_{nullptr};
  torch::nn::Linear condition_{nullptr};
  torch::nn::Conv2d policy1_{nullptr}, policy2_{nullptr};
  torch::nn::Linear value1_{nullptr}, value2_{nullptr};
};
TORCH_MODULE(PolicyNet);

// Illegal actions (mask == false) get a large negative logit.
torch::Tensor maskLogits(const torch::Tensor& logits, const torch::Tensor& mask);

}  // namespace polypack::train
