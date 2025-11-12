#pragma once
#include <torch/extension.h>

namespace cuda_op {

  torch::Tensor
  torch_flash_attn(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V);
} // namespace cuda_op
