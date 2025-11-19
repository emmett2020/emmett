#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  // We'll do reduction through difference thread blocks and use thread fence to make counter visible to all thread blocks.

  /// WARN: We're not validate this function.

  __device__ unsigned count = 0;
  __shared__ bool is_last_block_done;

  __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int s = 16; s > 0; s /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
  }

  __device__ float block_reduce_sum(float val, float* smem) {
    const unsigned tid       = threadIdx.x;
    const unsigned lid       = threadIdx.x % 32;
    const unsigned wid       = threadIdx.x / 32;
    const unsigned num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lid == 0) {
      smem[wid] = val;
    }
    __syncthreads();

    val = tid < num_warps ? smem[lid] : 0.F;
    if (wid == 0) {
      val = warp_reduce_sum(val);
    }
    return val;
  }

  // There are only one warp in one thread block.
  __global__ void sum(const float* A, unsigned N, volatile float* result) {
    __shared__ float smem[32];

    const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    float v = A[idx];
    v       = block_reduce_sum(v, smem);

    if (threadIdx.x == 0) {
      result[blockIdx.x] = v;
      __threadfence();

      unsigned int value = atomicInc(&count, gridDim.x);
      is_last_block_done = (value == (gridDim.x - 1));
    }
    __syncthreads();

    if (is_last_block_done) {
      const unsigned new_idx = threadIdx.x;
      if (new_idx < N) {
        float v         = result[new_idx];
        float total_sum = block_reduce_sum(v, smem);
        if (threadIdx.x == 0) {
          result[0] = total_sum;
          count     = 0;
        }
      }
    }
  }

} // namespace

int main() {
  float* ptr = nullptr;
  size_t N   = 1'024 * 1'024 * 1'024;
  cudaMalloc(&ptr, N * 4);
  size_t grid_size = (N + 127) / 128;
  cudaDeviceSynchronize();
  return 0;
}

