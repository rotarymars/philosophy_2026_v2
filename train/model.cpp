#include "model.hpp"

namespace polypack::train {

namespace F = torch::nn::functional;

PolicyNetImpl::PolicyNetImpl(const ModelConfig& c) : config_(c) {
  const int ch = c.channels;
  conv1_ = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, ch / 2, 3).padding(1)));
  conv2_ = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch / 2, ch, 3).stride(2).padding(1)));
  conv3_ = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(1)));
  conv4_ = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(1)));
  condition_ = register_module("condition", torch::nn::Linear(c.slotFeatures + c.globalFeatures, ch));
  policy1_ = register_module("policy1", torch::nn::Conv1d(torch::nn::Conv1dOptions(ch, ch, 3).padding(1)));
  policy2_ = register_module("policy2", torch::nn::Conv1d(torch::nn::Conv1dOptions(ch, 1, 1)));
  value1_ = register_module("value1", torch::nn::Linear(2 * ch + c.globalFeatures, 128));
  value2_ = register_module("value2", torch::nn::Linear(128, 1));

  // Start close to a uniform policy over legal actions.
  torch::NoGradGuard noGrad;
  policy2_->weight.mul_(0.01);
  policy2_->bias.zero_();
}

PolicyOutput PolicyNetImpl::forward(const torch::Tensor& binImages, const torch::Tensor& itemImages,
                                    const torch::Tensor& slotFeatures,
                                    const torch::Tensor& globalFeatures) {
  const int64_t n = binImages.size(0);
  const int64_t s = config_.slots;
  const int64_t r = config_.rotations;
  const int64_t h = config_.rasterSize;
  const int64_t w = config_.rasterSize;
  const int64_t m = n * s * r;

  const torch::Tensor bins = binImages.to(torch::kFloat).div(255.0).unsqueeze(2).expand({n, s, r, h, w});
  const torch::Tensor items = itemImages.to(torch::kFloat).div(255.0).unsqueeze(1).expand({n, s, r, h, w});
  torch::Tensor x = torch::stack({bins, items}, 3).reshape({m, 2, h, w});

  x = torch::relu(conv1_(x));
  x = torch::relu(conv2_(x));
  x = torch::relu(conv3_(x));

  const torch::Tensor slotF = slotFeatures.unsqueeze(2).expand({n, s, r, config_.slotFeatures});
  const torch::Tensor globalF =
      globalFeatures.view({n, 1, 1, config_.globalFeatures}).expand({n, s, r, config_.globalFeatures});
  const torch::Tensor cond = condition_(torch::cat({slotF, globalF}, 3).reshape({m, -1}));
  x = torch::relu(x + cond.view({m, -1, 1, 1}));
  x = torch::relu(conv4_(x));

  // Policy: one logit per entry column, read off the width axis.
  torch::Tensor p = x.mean(2);  // [m, C, W/2]
  p = torch::relu(policy1_(p));
  p = policy2_(p);  // [m, 1, W/2]
  p = F::interpolate(p, F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{config_.columns})
                            .mode(torch::kLinear)
                            .align_corners(true));
  torch::Tensor logits = p.reshape({n, s * r * config_.columns});

  // Value: pool over all (slot, rotation) pairs.
  const torch::Tensor pooled = x.mean({2, 3}).view({n, s * r, -1});
  const torch::Tensor v = torch::cat(
      {pooled.mean(1), std::get<0>(pooled.max(1)), globalFeatures}, 1);
  torch::Tensor value = value2_(torch::relu(value1_(v))).squeeze(-1);

  return {logits, value};
}

torch::Tensor maskLogits(const torch::Tensor& logits, const torch::Tensor& mask) {
  return logits.masked_fill(mask.logical_not(), -1e9);
}

}  // namespace polypack::train
