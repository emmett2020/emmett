#include <cstddef>
#include <iostream>
#include <cstdint>

namespace {

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

  const unsigned grid_size = (N + 127) / 128;
  normal<<<grid_size, 128>>>(a_ptr, N, b_ptr);
  bypass<<<grid_size, 128>>>(a_ptr, N, b_ptr);
}
