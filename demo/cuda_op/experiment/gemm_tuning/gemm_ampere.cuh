#include <cstddef>
#include <iostream>
#include <cstdint>

#include "gemm_base.cuh"

namespace {

  // B is col_major
  __global__ void
  gemm(const float* A,
       const float* B,
       unsigned M,
       unsigned N,
       unsigned K,
       unsigned tile_k,
       float* C) {
    unsigned row = (blockIdx.y * gridDim.y) + threadIdx.y;
    unsigned col = (blockIdx.x * gridDim.x) + threadIdx.x;

    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + blockDim.y * tile_k;

    float sum = 0.0F;
    for (unsigned k_base = 0; k_base < K; k_base += tile_k) {
      for (unsigned k_iter = 0; k_iter < tile_k; ++k_iter) {
        unsigned k = k_base + k_iter;
        if (k > K) {
          break;
        }
        As[static_cast<size_t>((threadIdx.y * tile_k) + k_iter)] = A[(row * K) + k];
        Bs[static_cast<size_t>((threadIdx.x * tile_k) + k_iter)] = B[(k * N) + col];
      }
      __syncthreads();

      for (int k_iter = 0; k_iter < tile_k; ++k_iter) {
        unsigned k = k_base + k_iter;
        if (k_iter > K) {
          break;
        }
        sum += As[(threadIdx.y * tile_k) + k_iter] * Bs[(threadIdx.x * tile_k) + k_iter];
      }
      __syncthreads();
    }

    C[(row * N) + col] = sum;
  }

  void launch_gemm(const float* A, const float* B, unsigned M, unsigned N, unsigned K, float* C) {
    constexpr uint32_t tile_m = 32;
    constexpr uint32_t tile_n = 32;
    constexpr uint32_t tile_k = 16;
    auto grid_dim             = dim3{(N + tile_n - 1) / tile_n, (M + tile_m - 1) / tile_m};
    auto blk_dim              = dim3{tile_n, tile_m};
    auto smem_size            = (tile_m * tile_k + tile_k * tile_n) * sizeof(float);
    gemm<<<grid_dim, blk_dim, smem_size>>>(A, B, M, N, K, tile_k, C);
  }

} // namespace

