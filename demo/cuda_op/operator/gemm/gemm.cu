#include "gemm/gemm.h"

#include <torch/extension.h>

#include "common/utils.h"

namespace cuda_op {

  constexpr std::size_t tile_size = 16;

  /// TODO: not valid
  __global__ void
  gemm(const float* a_ptr, const float* b_ptr, unsigned M, unsigned N, unsigned K, float* c_ptr) {
    constexpr std::size_t tile_k = 16;
    __shared__ float As[tile_size][tile_k];
    __shared__ float Bs[tile_size][tile_k];

    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
      return;
    }

    float sum = 0.F;
    for (int k_base = 0; k_base < K; k_base += tile_k) {
      for (int k_iter = 0; k_iter < tile_k; ++k_iter) {
        int k = k_base + k_iter;
        if (k > K) {
          break;
        }

        As[threadIdx.y][k] = a_ptr[row * K + k];
        Bs[threadIdx.x][k] = b_ptr[col * K + k];
        __syncthreads();
      }
      for (int k_iter = 0; k_iter < tile_k; ++k_iter) {
        int k = k_base + k_iter;
        if (k > K) {
          break;
        }
        float a = As[threadIdx.y][k];
        float b = Bs[threadIdx.x][k];

        sum += a * b;
      }
    }

    c_ptr[row * N + col] = sum;
  }

  cudaError_t launch_gemm(
    const float* a_ptr,
    const float* b_ptr,
    unsigned M,
    unsigned N,
    unsigned K,
    float* c_ptr) {
    const dim3 blk_size{tile_size, tile_size};
    const dim3 grid_size{static_cast<unsigned int>((M + tile_size - 1) / tile_size),
                         static_cast<unsigned int>((N + tile_size - 1) / tile_size)};
    gemm<<<grid_size, blk_size>>>(a_ptr, b_ptr, M, N, K, c_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_gemm(const torch::Tensor& A, const torch::Tensor& B) {
    const auto a_shape = A.sizes();
    const auto b_shape = B.sizes();
    TORCH_CHECK(a_shape.size() == 2, "shape size of a must be 2");
    TORCH_CHECK(b_shape.size() == 2, "shape size of b must be 2");
    TORCH_CHECK(a_shape[1] == b_shape[1], "K dimension doesn't same");
    const int M = a_shape[0];
    const int K = a_shape[1];
    const int N = b_shape[0];

    const float* a_ptr = A.data_ptr<float>();
    const float* b_ptr = B.data_ptr<float>();
    auto output        = torch::zeros({M, N}, A.options());
    float* output_ptr  = output.data_ptr<float>();
    cuda_check(launch_gemm(a_ptr, b_ptr, M, N, K, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
