#pragma once

#include <cassert>
#include <cstddef>

#include "gemm_base.cuh"

/// Most of kernels in this file have worse performance. However it doesn't
/// matters since we only use theses kernels to illustrate how task splitting
/// works in GEMM.

namespace {
  /// Grid has one dim.  Per thread block per (tile_m * K) of A, (completed B * tile_m) of B, (tile_m * N) of C.
  /// Block has one dim. Per thread per row (1 * K) of A, completed B, N of C.
  /// Without shared memory or other optimizations.
  __global__ void
  gemm_split_m(const float* A, const float* B, unsigned K, unsigned N, unsigned tile_m, float* C) {
    unsigned a_blk_row = blockIdx.x * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.x;

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
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");

    auto grid_dim = dim3{M / tile_m};
    auto blk_dim  = dim3{tile_m};
    gemm_split_m<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, C);
  }

  __global__ void gemm_split_m_blk2dims(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
    unsigned a_blk_row = blockIdx.x * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.y;

    for (int x = 0; x < N; ++x) {
      unsigned k   = threadIdx.x;
      float val    = A[(a_row * K) + k] * B[(k * N) + x];
      float* c_ptr = C + (static_cast<size_t>(a_row * N)) + x;
      atomicAdd(c_ptr, val);
    }
  }

  void launch_gemm_split_m_blk2dims(
    const float* A,
    const float* B,
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");

    auto grid_dim = dim3{M / tile_m};
    auto blk_dim  = dim3{K, tile_m};
    gemm_split_m_blk2dims<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, C);
  }

} // namespace
