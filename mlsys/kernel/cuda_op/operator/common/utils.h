#pragma once

#include <stdexcept>
#include <format>

#include <torch/extension.h>

namespace cuda_op {
  inline void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      auto msg = std::format("[CUDA ERROR] code: {}, str: {}",
                             static_cast<int>(err),
                             cudaGetErrorString(err));
      throw std::runtime_error(msg);
    }
  }
} // namespace cuda_op
