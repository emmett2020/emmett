#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon);

  torch::Tensor torch_layer_norm_nlp(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon);
} // namespace cuda_op
