#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_add(const torch::Tensor& a, const torch::Tensor& b);
} // namespace cuda_op

