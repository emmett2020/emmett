#pragma once

#include <torch/extension.h>

namespace cuda_op {
  torch::Tensor torch_rms_norm(const torch::Tensor& input, float epsilon);
} // namespace cuda_op
