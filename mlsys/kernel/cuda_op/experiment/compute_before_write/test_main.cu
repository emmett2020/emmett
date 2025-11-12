#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  __global__ void slightly_compute(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      float val = rsqrtf(static_cast<float>(idx));
      ptr[idx]  = val;
    }
  }

  __global__ void heavily_compute(float* ptr, unsigned N) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      float val = rsqrtf(static_cast<float>(idx));
      val       = val * rsqrtf(val + 2);
      val       = val * rsqrtf(val * val);
      val       = val * rsqrtf(val / 2 * val);
      for (int i = 0; i < 3; ++i) {
        val  = val * sqrtf(val) + 1;
        val /= (sqrtf(val) + 0.143);
      }
      ptr[idx] = val;
    }
  }


} // namespace

int main() {
  float* ptr = nullptr;
  size_t N   = 1'024 * 1'024 * 1'024;
  cudaMalloc(&ptr, N * 4);
  size_t grid_size = (N + 127) / 128;
  slightly_compute<<<grid_size, 128>>>(ptr, N);
  heavily_compute<<<grid_size, 128>>>(ptr, N);
  cudaDeviceSynchronize();
  return 0;
}

