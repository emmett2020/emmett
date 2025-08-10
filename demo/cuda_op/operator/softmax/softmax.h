#pragma once
#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_softmax(const torch::Tensor& input);

  torch::Tensor torch_safe_softmax(const torch::Tensor& input);
} // namespace cuda_op
