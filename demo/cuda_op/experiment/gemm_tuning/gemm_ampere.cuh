#include <cstddef>
#include <iostream>
#include <cstdint>

#include "gemm_base.cuh"

namespace {

  constexpr std::size_t tile_m = 32;
  constexpr std::size_t tile_n = 32;
  constexpr std::size_t tile_k = 32;

  __global__ void
  gemm(const float* A, const float* B, unsigned M, unsigned N, unsigned K, float* C) {
    __shared__ float As[tile_m][tile_k];
    __shared__ float Bs[tile_k][tile_n];

    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;

    const unsigned row = by * tile_m + ty;
    const unsigned col = bx * tile_n + tx;

    float sum        = 0.0F;
    const unsigned T = (K + tile_k - 1) / tile_k;
    for (int t = 0; t < T; ++t) {
      unsigned a_col = t * tile_k + tx;
      if (row < M && a_col < K) {
        As[ty][tx] = A[row * K + a_col];
      } else {
        As[ty][tx] = 0.0F;
      }

      if (t * tile_k + ty < K && col < N) {
        Bs[ty][tx] = B[(t * tile_k + ty) * N + col];
      } else {
        Bs[ty][tx] = 0.0F;
      }

      __syncthreads();
      for (int k = 0; k < tile_k; ++k) {
        sum += As[ty][k] * Bs[k][tx];
      }
      __syncthreads();
    }

    if (row < M && col < N) {
      C[row * N + col] = sum;
    }
  }

  void launch_gemm(const float* A, const float* B, unsigned M, unsigned N, unsigned K, float* C) {
    auto grid_dim = dim3{static_cast<unsigned int>((N + tile_n - 1) / tile_n),
                         static_cast<unsigned int>((M + tile_m - 1) / tile_m)};
    auto blk_dim  = dim3{tile_n, tile_m};
    gemm<<<grid_dim, blk_dim>>>(A, B, M, N, K, C);
  }

} // namespace

