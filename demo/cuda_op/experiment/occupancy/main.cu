#include <cuda_runtime.h>
#include <iostream>

namespace {

  __global__ void TheroticalOccupancy(float* d_out, int ele) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ele) {
      d_out[threadIdx.x] = 0;
    }
  }
} // namespace

int main() {
  const std::size_t size = 1 * 1'024 * 1'024;
  float* buffer          = nullptr;
  cudaMalloc(&buffer, size);
  TheroticalOccupancy<<<1, 144>>>(buffer, size / 4);
  return 0;
}
