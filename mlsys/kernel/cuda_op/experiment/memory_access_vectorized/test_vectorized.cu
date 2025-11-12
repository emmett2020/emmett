#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {

  /// NOTE:
  /// To simplify, N should be divisible by 4

  __global__ void per_thread_1_element_grid_stride(float* ptr, unsigned N) {
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
      ptr[idx] = static_cast<float>(idx);
    }
  }

  __global__ void per_thread_4_elements_grid_stride(float* ptr, unsigned N) {
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x * 4;
    size_t stride = gridDim.x * blockDim.x * 4;
    for (size_t i = idx; i < N; i += stride) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ptr[idx + j] = static_cast<float>(idx + j);
      }
    }
  }

  __global__ void per_thread_4_elements_vectorized(float* ptr, unsigned N) {
    size_t idx        = blockIdx.x * blockDim.x + threadIdx.x;
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
    size_t stride     = gridDim.x * blockDim.x * 4;
    for (size_t i = global_idx; i < N; i += stride) {
      auto* ptr4 = reinterpret_cast<float4*>(ptr);
      auto data = make_float4(global_idx, global_idx + 1, global_idx + 2, global_idx + 3); // NOLINT
      ptr4[idx] = data;
    }
  }


} // namespace

int main() {
  float* ptr = nullptr;
  size_t N   = 1'024 * 1'024 * 1'024;
  cudaMalloc(&ptr, N * 4);

  {
    size_t grid_size = (N + 127) / 128;
    per_thread_1_element_grid_stride<<<grid_size, 128>>>(ptr, N);
  }

  {
    size_t vectorized_n = N / 4;
    size_t grid_size    = (vectorized_n + 127) / 128;
    per_thread_4_elements_vectorized<<<grid_size, 128>>>(ptr, N);
  }


  {
    size_t vectorized_n = N / 4;
    size_t grid_size    = (vectorized_n + 127) / 128;
    per_thread_4_elements_grid_stride<<<grid_size, 128>>>(ptr, N);
  }

  cudaDeviceSynchronize();
  return 0;
}

