#pragma once
#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_group_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int num_groups,
    float epsilon);
} // namespace cuda_op
