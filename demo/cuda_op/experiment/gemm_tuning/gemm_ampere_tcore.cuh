#include <cstddef>
#include <iostream>
#include <cstdint>

#include <cuda_fp16.h>
#include <mma.h>

#include "gemm_base.cuh"

using namespace nvcuda; // NOLINT

namespace {
  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, half* C) {
  }

  void launch_gemm_ampere_tcore(
    const float* A,
    const float* B,
    unsigned M,
    unsigned N,
    unsigned K,
    float* C) {
    auto grid_dim = dim3{static_cast<unsigned int>((N + tile - 1) / tile),
                         static_cast<unsigned int>((M + tile - 1) / tile)};
  }

} // namespace

