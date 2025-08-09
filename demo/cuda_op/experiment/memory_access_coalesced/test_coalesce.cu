#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  __global__ void coalesced_access(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      ptr[idx] = static_cast<float>(idx);
    }
  }

  __global__ void unordered_coalesced_access(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    idx = idx % 2 == 0 ? idx + 1 : idx - 1;
    if (idx < N) {
      ptr[idx] = static_cast<float>(idx);
    }
  }

  // | w0t0 | w1t0 | w2t0 | w3t0 | w0t1 | w1t1 | w2t1 | w3t1 | ...
  __global__ void non_coalesced_access(float* ptr, unsigned N) {
    unsigned num_warps = threadIdx.x / 32;
    unsigned warp_idx  = threadIdx.x % 32;
    unsigned idx       = blockIdx.x * blockDim.x + threadIdx.x * num_warps + warp_idx;

    if (idx < N) {
      ptr[idx] = static_cast<float>(idx);
    }
  }


} // namespace

int main() {
  float* ptr = nullptr;
  size_t N   = 1'024 * 1'024 * 1'024;
  cudaMalloc(&ptr, N * 4);
  size_t grid_size = (N + 127) / 128;
  coalesced_access<<<grid_size, 128>>>(ptr, N);
  unordered_coalesced_access<<<grid_size, 128>>>(ptr, N);
  non_coalesced_access<<<grid_size, 128>>>(ptr, N);
  cudaDeviceSynchronize();
  return 0;
}

