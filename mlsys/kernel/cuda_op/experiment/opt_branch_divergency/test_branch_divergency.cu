#include <cuda_runtime.h>
#include <cuda.h>

namespace {
  __global__ void branch_predication_before(float* data, float threshold) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (data[idx] > threshold) {
      data[idx] = sinf(data[idx]);
    }
  }

  __global__ void branch_predication_after(float* data, float threshold) {
    unsigned idx     = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;
    unsigned mask    = __ballot_sync(0xFFFFFFFF, static_cast<int>(data[idx] > threshold));

    if ((mask & (1 << lane_id)) != 0U) {
      float val = data[idx];
      data[idx] = sinf(val);
    }
  }

  __global__ void likely_before(float* data, int threshold) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < threshold) {
      data[idx] = sinf(data[idx]);
    } else {
      data[idx] = cosf(data[idx]);
    }
  }

  __global__ void likely_after(float* data, int threshold) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < threshold) {
      data[idx] = sinf(data[idx]);
    } else [[likely]] {
      data[idx] = cosf(data[idx]);
    }
  }

} // namespace

int main() {
  const std::size_t N      = 1'024 * 1'024 * 1'024;
  const std::size_t n_size = N * sizeof(float);

  float* ptr = nullptr;
  cudaMalloc(&ptr, n_size);

  const std::size_t grid = (N + 1'023) / 1'024;
  // branch_predication_before<<<grid, 1'024>>>(ptr, 2.0);
  // branch_predication_after<<<grid, 1'024>>>(ptr, 2.0);

  likely_before<<<grid, 1'024>>>(ptr, 2);
  likely_after<<<grid, 1'024>>>(ptr, 2);

  cudaDeviceSynchronize();
  return 0;
}

