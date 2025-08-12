#pragma once
#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_gemm(const torch::Tensor& A, const torch::Tensor& B);
} // namespace cuda_op
