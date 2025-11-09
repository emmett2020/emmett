#include <cuda_runtime.h>
#include <iostream>

namespace {

  __global__ void TheoreticalOccupancy(float* d_out, int ele) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ele) {
      d_out[threadIdx.x] = 0;
    }
  }

  __global__ void AchievedOccupancy(float* d_out, int ele) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ele) {
      d_out[threadIdx.x] = 0;
    }
  }

  __global__ void AchievedOccupancyCompute(float* d_out, int ele) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ele) {
      float a = sinf(idx);
    }
  }

} // namespace

int main() {
  const std::size_t size = 1 * 1'024U * 1'024 * 1'024;
  float* buffer          = nullptr;
  cudaMalloc(&buffer, size);
  // In 4070, max active_warps is 48, that's to say, if we could reach up to 48
  // warps, we'll get 100% theoretical occupancy.
  // We use two block since one block only support 1024 threads in 4070.
  AchievedOccupancy<<<2, 24 * 32>>>(buffer, size / 4);

  AchievedOccupancy<<<2, 24 * 32>>>(buffer, size / 4);

  AchievedOccupancy<<<24, 24 * 32>>>(buffer, size / 4);

  // User inputed block counter doesn't affect active blocks.
  AchievedOccupancy<<<128, 24 * 32>>>(buffer, size / 4);

  AchievedOccupancyCompute<<<128, 24 * 32>>>(buffer, size / 4);
  return 0;
}
