#pragma once

#include <cassert>

namespace {
  /// Grid has one dim.  Per thread block per (tile_m * K) of A, (completed B * tile_m) of B, (tile_m * N) of C.
  /// Block has one dim. Per thread per row (1 * K) of A, completed B, N of C.
  /// Without shared memory or other optimizations.
  /// It has worse performance.
  __global__ void
  gemm_split_m(const float* A, const float* B, int M, int K, int N, unsigned tile_m, float* C) {
    unsigned a_blk_row = blockIdx.x * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.x;
    if (a_row >= M) {
      return;
    }

    for (int x = 0; x < N; ++x) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[(a_row * K) + k] * B[(k * N) + x];
      }
      C[(a_row * N) + x] = sum;
    }
  }

  void launch_gemm_split_m(
    const float* A,
    const float* B,
    int M,
    int K,
    int N,
    unsigned tile_m,
    float* C) {
    auto grid_dim = dim3{((M + tile_m - 1) / tile_m)};
    auto blk_dim  = dim3{tile_m};
    gemm_split_m<<<grid_dim, blk_dim>>>(A, B, M, K, N, tile_m, C);
  }

} // namespace
