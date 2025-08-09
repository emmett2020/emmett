#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  __global__ void per_thread_1_element(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      ptr[idx] = static_cast<float>(idx);
    }
  }

  // To simplify, N should be divisible by 4
  __global__ void per_thread_4_elements(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= N) {
      return;
    }
    float4 data = make_float4(0, 1, 2, 3);
    auto* ptr4  = reinterpret_cast<float4*>(ptr);
    ptr4[idx]   = data;
  }


} // namespace

int main() {
  float* ptr = nullptr;
  size_t N   = 1'024 * 1'024 * 1'024;
  cudaMalloc(&ptr, N * 4);

  {
    size_t vectorized_n = (N + 4 - 1) / 4;
    size_t grid_size    = (vectorized_n + 127) / 128;
    per_thread_4_elements<<<grid_size, 128>>>(ptr, N);
  }

  {
    size_t grid_size = (N + 127) / 128;
    per_thread_1_element<<<grid_size, 128>>>(ptr, N);
  }

  cudaDeviceSynchronize();
  return 0;
}

