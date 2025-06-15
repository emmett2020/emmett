#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    torch::Tensor& running_mean,
    torch::Tensor& running_var,
    float epsilon,
    float momentum,
    bool training);
} // namespace cuda_op
