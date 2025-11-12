#include <iostream>
#include <cstdint>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  // One dimension
  __global__ void bank_conflict(float* data, int N) {
    __shared__ float smem[64];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    // Each bank is visited by two lane.
    // e.g.:
    // lane0,  lane16 -> bank0
    // lane1,  lane17 -> bank1
    // lane15, lane31 -> bank15
    unsigned bank_idx            = lane_id % 16;
    unsigned bank_element_offset = bank_idx;

    // e.g.:
    // bank0 is visited by lane0 and lane16,
    // lane0  visit the beginning of bank0
    // lane16 visit bank0 + 128 bytes
    unsigned sub_bank_idx            = lane_id / 16;
    unsigned sub_bank_element_offset = sub_bank_idx * 32;
    unsigned s_element_idx           = bank_element_offset + sub_bank_element_offset;

    smem[s_element_idx] = sinf(static_cast<float>(lane_id));
    __syncthreads();

    if (tid < N) {
      data[tid] = smem[s_element_idx];
    }
  }

  // Two dimensions
  __global__ void bank_conflict2(float* data) {
    __shared__ float smem[32][32];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (int i = 0; i < 32; ++i) {
      smem[lane_id][i] = 0;
    }
    __syncthreads();

    for (int i = 0; i < 32; ++i) {
      data[tid] = smem[lane_id][i];
    }
  }

  // Swap thread dimensions will fix bank conflict.
  __global__ void permute(float* data) {
    __shared__ float smem[32][32];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (auto& j: smem) {
      j[lane_id] = 0;
    }
    __syncthreads();

    for (auto& j: smem) {
      data[tid] = j[lane_id];
    }
  }

  __global__ void memory_padding(float* data) {
    __shared__ float smem[32][33];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (int i = 0; i < 32; ++i) {
      smem[lane_id][i] = 0;
    }
    __syncthreads();

    for (int i = 0; i < 32; ++i) {
      data[tid] = smem[lane_id][i];
    }
  }
} // namespace

auto main() noexcept(false) -> int {
  const int N            = 1 * 1'024 * 1'024;
  const std::size_t size = N * sizeof(float);
  float* buffer          = nullptr;
  cuda_check(cudaMalloc(&buffer, size));

  bank_conflict2<<<1, 32>>>(buffer);
  permute<<<1, 32>>>(buffer);
  memory_padding<<<1, 32>>>(buffer);

  // Recycle resources
  cudaFree(buffer);
}
