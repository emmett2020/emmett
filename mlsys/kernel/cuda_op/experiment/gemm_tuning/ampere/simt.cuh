#include <cstddef>
#include <iostream>
#include <cstdint>

#include "gemm_base.cuh"

namespace {

  constexpr std::size_t tile = 16;

  __global__ void
  gemm(const float* A, const float* B, unsigned M, unsigned N, unsigned K, float* C) {
    __shared__ float As[tile][tile];
    __shared__ float Bs[tile][tile];

    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;

    const unsigned row = by * tile + ty;
    const unsigned col = bx * tile + tx;

    float sum        = 0.0F;
    const unsigned T = (K + tile - 1) / tile;
    for (int t = 0; t < T; ++t) {
      unsigned a_col = t * tile + tx;
      if (row < M && a_col < K) {
        As[ty][tx] = A[row * K + a_col];
      } else {
        As[ty][tx] = 0.0F;
      }

      unsigned b_row = t * tile + ty;
      if (b_row < K && col < N) {
        Bs[ty][tx] = B[b_row * N + col];
      } else {
        Bs[ty][tx] = 0.0F;
      }

      __syncthreads();
      for (int k = 0; k < tile; ++k) {
        sum += As[ty][k] * Bs[k][tx];
      }
      __syncthreads();
    }

    if (row < M && col < N) {
      C[row * N + col] = sum;
    }
  }

  void launch_gemm(const float* A, const float* B, unsigned M, unsigned N, unsigned K, float* C) {
    auto grid_dim = dim3{static_cast<unsigned int>((N + tile - 1) / tile),
                         static_cast<unsigned int>((M + tile - 1) / tile)};
    auto blk_dim  = dim3{tile, tile};
    gemm<<<grid_dim, blk_dim>>>(A, B, M, N, K, C);
  }

} // namespace

