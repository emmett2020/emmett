#pragma once
#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_conv2d(const torch::Tensor& input, const torch::Tensor& kernel);
} // namespace cuda_op
