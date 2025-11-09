#pragma once

#include <cassert>
#include <cstddef>

#include "gemm_base.cuh"

/// Most of kernels in this file have worse performance. However it doesn't
/// matters since we only use theses kernels to illustrate how GEMM task
/// splitting works.

namespace {
  /// Grid has one dim.  Per thread block per (tile_m * K) of A, (completed B * tile_m) of B, (tile_m * N) of C.
  /// Block has one dim. Per thread per row (1 * K) of A, completed B, N of C.
  /// Without shared memory or other optimizations.
  __global__ void gemm_split_m_grid_1dim(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
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

  void launch_gemm_split_m_grid_1dim(
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
    gemm_split_m_grid_1dim<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, C);
  }

  __global__ void gemm_split_m_n_grid_1dim_blk_2dims(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
    unsigned a_blk_row = blockIdx.x * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.y;
    unsigned b_col     = threadIdx.x;

    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[(a_row * K) + k] * B[(k * N) + b_col];
    }
    C[(a_row * N) + b_col] = sum;
  }

  void launch_gemm_split_m_n_grid_1dim_blk_2dims(
    const float* A,
    const float* B,
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");
    throw_if(N >= 1'024, "unsupported");

    auto grid_dim = dim3{M / tile_m};
    auto blk_dim  = dim3{N, tile_m};
    gemm_split_m_n_grid_1dim_blk_2dims<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, C);
  }

  /// Grid has two dim.  Per thread block per (tile_m * K) of A, (K * tile_n * tile_m) of B, (tile_m * tile_n) of C.
  /// Block has one dim. Per thread per row (1 * K) of A, per tile (K * tile_n) of B, per tile_n of C.
  /// Without shared memory or other optimizations.
  __global__ void gemm_split_m_grid_2dims(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_n,
    float* C) {
    unsigned a_blk_row = blockIdx.y * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.x;

    unsigned b_blk_col = blockIdx.x * tile_n;

    /// NOTE: Thread block in same y do have load totally consistent group of
    /// contiguous rows, however, without the helping of shared memory or etc, we
    /// still need to travel the tile_n to make sure not missing some columns.
    for (int y = 0; y < tile_n; ++y) {
      unsigned b_col = b_blk_col + y;
      float sum      = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[(a_row * K) + k] * B[(k * N) + b_col];
      }
      C[(a_row * N) + b_col] = sum;
    }
  }

  void launch_gemm_split_m_grid_2dims(
    const float* A,
    const float* B,
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_n,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");
    throw_if(N % tile_n != 0, "unsupported");

    auto grid_dim = dim3{N / tile_n, M / tile_m};
    auto blk_dim  = dim3{tile_m};
    gemm_split_m_grid_2dims<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, tile_n, C);
  }

  /// Grid has two dim.  Per thread block per (tile_m * K) of A, (K * tile_n) of B, (tile_m * tile_n) of C.
  /// Block has one dim. Per thread per row (1 * K) of A, per column (K * 1) of B, per element of C.
  /// Still has various limitations.
  __global__ void gemm_split_m_grid_2dims_blk_2dims_shared(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_n,
    float* C) {
    unsigned a_blk_row = blockIdx.y * tile_m;
    unsigned a_row     = a_blk_row + threadIdx.y;

    unsigned b_blk_col = blockIdx.x * tile_n;
    unsigned b_col     = b_blk_col + threadIdx.x;

    // Assume tile_m <= 32, tile_n <= 32 and k <= 64
    __shared__ float As[32][64];
    __shared__ float Bs[64][32];
    for (unsigned k_base = 0; k_base < K; k_base += blockDim.x) {
      unsigned a_col = k_base + threadIdx.x;
      if (a_col < K) {
        As[threadIdx.y][a_col] = A[(a_row * K) + a_col];
      }
    }
    for (unsigned k_base = 0; k_base < K; k_base += blockDim.y) {
      unsigned b_row = k_base + threadIdx.y;
      if (b_row < K) {
        Bs[b_row][threadIdx.x] = B[(b_row * N) + b_col];
      }
    }
    __syncthreads();

    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    C[(a_row * N) + b_col] = sum;
  }

  void launch_gemm_split_m_grid_2dims_blk_2dims_shared(
    const float* A,
    const float* B,
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_n,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");
    throw_if(N % tile_n != 0, "unsupported");
    throw_if(tile_m > 32, "unsupported");
    throw_if(tile_n > 32, "unsupported");
    throw_if(K > 64, "unsupported");

    auto grid_dim = dim3{N / tile_n, M / tile_m};
    auto blk_dim  = dim3{tile_n, tile_m};
    gemm_split_m_grid_2dims_blk_2dims_shared<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, tile_n, C);
  }

  __global__ void gemm_split_m_k_grid_1dim_blk_2dims(
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

  void launch_gemm_split_m_k_grid_1dim_blk_2dims(
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
    gemm_split_m_k_grid_1dim_blk_2dims<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, C);
  }

  __global__ void gemm_split_m_k_grid_2dims(
    const float* A,
    const float* B,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_k,
    float* C) {
    unsigned a_block_row = blockIdx.y * tile_m;
    unsigned a_block_col = blockIdx.x * tile_k;
    unsigned a_row       = a_block_row + threadIdx.x;

    unsigned b_block_row = blockIdx.x * tile_k;

    for (int y = 0; y < N; ++y) {
      unsigned b_col = y;

      float sum = 0;
      for (int x = 0; x < tile_k; ++x) {
        unsigned a_col  = a_block_col + x;
        unsigned b_row  = b_block_row + x;
        sum            += A[(a_row * K) + a_col] * B[(b_row * N) + b_col];
      }
      float* ptr = C + (static_cast<size_t>(a_row * N)) + b_col;
      atomicAdd(ptr, sum);
    }
  }

  void launch_gemm_split_m_k_grid_2dim(
    const float* A,
    const float* B,
    unsigned M,
    unsigned K,
    unsigned N,
    unsigned tile_m,
    unsigned tile_k,
    float* C) {
    throw_if(M % tile_m != 0, "unsupported");
    throw_if(K % tile_k != 0, "unsupported");

    auto grid_dim = dim3{K / tile_k, M / tile_m};
    auto blk_dim  = dim3{tile_m};
    gemm_split_m_k_grid_2dims<<<grid_dim, blk_dim>>>(A, B, K, N, tile_m, tile_k, C);
  }

  __global__ void simple_matmul(const float* A, const float* B, int M, int K, int N, float* C) {
    unsigned row = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < M && col < N) {
      float sum = 0.0F;

      for (int k = 0; k < K; k++) {
        sum += A[(row * K) + k] * B[(k * N) + col];
      }

      C[(row * N) + col] = sum;
    }
  }


} // namespace
