#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  // Undefined behavior
  __global__ void dead_lock_sync_in_condition() {
    unsigned tid = threadIdx.x;
    if (tid % 2 == 0) {
      __syncthreads();
    }
  }

} // namespace

int main() {
  dead_lock_sync_in_condition<<<1, 1'024>>>();
  cudaDeviceSynchronize();
  return 0;
}

