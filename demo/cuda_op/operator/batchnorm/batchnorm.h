#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_batch_norm(
    const torch::Tensor& input,
    torch::Tensor& running_mean,
    torch::Tensor& running_var,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon,
    float momentum,
    bool training);
} // namespace cuda_op
