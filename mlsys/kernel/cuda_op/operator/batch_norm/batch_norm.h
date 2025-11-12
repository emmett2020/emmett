#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon);

} // namespace cuda_op
