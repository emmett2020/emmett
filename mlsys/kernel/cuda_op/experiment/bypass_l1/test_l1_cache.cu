#include <cstddef>
#include <iostream>
#include <cstdint>

namespace {
  __global__ void transpose(const float* A, const int N, float* B) {
    const unsigned nx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N) {
      B[ny * N + nx] = A[nx * N + ny];
    }
  }

  __global__ void transpose_bypass(const float* A, const int N, float* B) {
    const unsigned nx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N) {
      B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
  }

  __global__ void normal(const float* input, unsigned N, float* output) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned i = idx; i < N; i += gridDim.x * blockDim.x) {
      float value = input[idx];
      output[idx] = value;
    }
  }

  __global__ void bypass(const float* input, unsigned N, float* output) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned i = idx; i < N; i += gridDim.x * blockDim.x) {
      float value = __ldg(input + idx);
      output[idx] = value;
    }
  }


} // namespace

auto main() noexcept(false) -> int {
  const unsigned N  = 4'096;
  const size_t size = static_cast<uint64_t>(N) * sizeof(float);

  float* a_ptr = nullptr;
  float* b_ptr = nullptr;
  cudaMalloc(&a_ptr, size);
  cudaMalloc(&b_ptr, size);

  cudaFuncSetCacheConfig(normal, cudaFuncCache::cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(bypass, cudaFuncCache::cudaFuncCachePreferL1);

  int grid_size = N / 128;
  normal<<<grid_size, 128>>>(a_ptr, N, b_ptr);
  bypass<<<grid_size, 128>>>(a_ptr, N, b_ptr);


  // const dim3 grid_size = {N / 32, N / 32};
  // const dim3 blk_size  = {32, 32};
  // transpose<<<grid_size, blk_size>>>(a_ptr, N, b_ptr);
  // transpose_bypass<<<grid_size, blk_size>>>(a_ptr, N, b_ptr);
}
