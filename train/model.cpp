#include "model.hpp"

namespace polypack::train {

namespace F = torch::nn::functional;

PolicyNetImpl::PolicyNetImpl(const ModelConfig& c) : config_(c) {
  const int ch = c.channels;
  const int summary = kItemSummary * kItemSummary;
  conv1_ = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, ch / 2, 3).padding(1)));
  // Stride 2, padded by hand (see forward) so output cell (i, j) is centred on
  // raster pixel (2i + 1, 2j): the lower-left pixel of grid spot
  // (gx = j, gy = G - 1 - i) when grid == rasterSize / 2.
  conv2_ = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch / 2, ch, 3).stride(2)));
  conv3_ = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch + 1, ch, 3).padding(1)));
  conv4_ = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(2).dilation(2)));
  condition_ = register_module("condition",
                               torch::nn::Linear(c.slotFeatures + c.globalFeatures + summary, ch));
  policy1_ = register_module("policy1", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, ch, 3).padding(1)));
  policy2_ = register_module("policy2", torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, 1, 1)));
  value1_ = register_module("value1", torch::nn::Linear(2 * ch + c.globalFeatures + c.criticFeatures, 128));
  value2_ = register_module("value2", torch::nn::Linear(128, 1));

  // Start close to a uniform policy over legal actions.
  torch::NoGradGuard noGrad;
  policy2_->weight.mul_(0.01);
  policy2_->bias.zero_();
}

PolicyOutput PolicyNetImpl::forward(const torch::Tensor& binImages, const torch::Tensor& itemImages,
                                    const torch::Tensor& contactImages,
                                    const torch::Tensor& slotFeatures,
                                    const torch::Tensor& globalFeatures, const torch::Tensor& mask,
                                    const torch::Tensor& criticFeatures) {
  const int64_t n = binImages.size(0);
  const int64_t s = config_.slots;
  const int64_t r = config_.rotations;
  const int64_t g = config_.grid;
  const int64_t h = config_.rasterSize;
  const int64_t w = config_.rasterSize;
  const int64_t m = n * s * r;
  const int64_t h2 = h / 2;
  const int64_t w2 = w / 2;

  // Raster-resolution input: bin occupancy and contact hint for each pair.
  const torch::Tensor bins = binImages.to(torch::kFloat).div(255.0).unsqueeze(2).expand({n, s, r, h, w});
  const torch::Tensor contact = contactImages.to(torch::kFloat).div(255.0).view({n, s, r, h, w});
  torch::Tensor x = torch::stack({bins, contact}, 3).reshape({m, 2, h, w});

  x = torch::relu(conv1_(x));
  x = F::pad(x, F::PadFuncOptions({1, 0, 0, 1}));  // left 1, right 0, top 0, bottom 1
  x = torch::relu(conv2_(x));                      // [m, C, h/2, w/2]

  // Exact legality of each spot, as a picture aligned with the features
  // (row 0 = top, so gy is flipped).
  torch::Tensor legal = mask.view({m, 1, g, g}).to(torch::kFloat).flip({2});
  if (g != h2 || g != w2) {
    legal = F::interpolate(legal, F::InterpolateFuncOptions()
                                      .size(std::vector<int64_t>{h2, w2})
                                      .mode(torch::kNearest));
  }

  // Per-pair conditioning: slot features, global features, item thumbnail.
  const torch::Tensor thumb =
      F::adaptive_avg_pool2d(itemImages.to(torch::kFloat).div(255.0),
                             F::AdaptiveAvgPool2dFuncOptions({kItemSummary, kItemSummary}))
          .view({n, 1, r, kItemSummary * kItemSummary})
          .expand({n, s, r, kItemSummary * kItemSummary});
  const torch::Tensor slotF = slotFeatures.unsqueeze(2).expand({n, s, r, config_.slotFeatures});
  const torch::Tensor globalF =
      globalFeatures.view({n, 1, 1, config_.globalFeatures}).expand({n, s, r, config_.globalFeatures});
  const torch::Tensor cond = condition_(torch::cat({slotF, globalF, thumb}, 3).reshape({m, -1}));

  x = torch::relu(conv3_(torch::cat({x, legal}, 1)) + cond.view({m, -1, 1, 1}));
  x = torch::relu(conv4_(x));

  // Policy: one logit per grid spot.
  torch::Tensor p = policy2_(torch::relu(policy1_(x)));  // [m, 1, h/2, w/2]
  if (g != h2 || g != w2) {
    p = F::interpolate(p, F::InterpolateFuncOptions()
                              .size(std::vector<int64_t>{g, g})
                              .mode(torch::kBilinear)
                              .align_corners(false));
  }
  const torch::Tensor logits = p.flip({2}).reshape({n, s * r * g * g});  // row index -> gy

  // Value: pool over all (slot, rotation) pairs.
  const torch::Tensor pooled = x.mean({2, 3}).view({n, s * r, -1});
  const torch::Tensor v =
      torch::cat({pooled.mean(1), std::get<0>(pooled.max(1)), globalFeatures, criticFeatures}, 1);
  torch::Tensor value = value2_(torch::relu(value1_(v))).squeeze(-1);

  return {logits, value};
}

torch::Tensor maskLogits(const torch::Tensor& logits, const torch::Tensor& mask) {
  return logits.masked_fill(mask.logical_not(), -1e9);
}

}  // namespace polypack::train
