#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_sigmoid(const torch::Tensor& input);
} // namespace cuda_op
