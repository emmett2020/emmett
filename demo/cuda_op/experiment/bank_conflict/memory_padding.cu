#include <iostream>
#include <cstdint>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  __global__ void bank_conflict(float* data, int N) {
    __shared__ float smem[32];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    // Each bank is visited by two lane.
    // e.g.:
    // lane0,  lane16 -> bank0
    // lane1,  lane17 -> bank1
    // lane15, lane31 -> bank15
    unsigned bank_idx            = lane_id % 16;
    unsigned bank_element_offset = bank_idx * 4;

    // e.g.:
    // bank0 is visited by lane0 and lane16,
    // lane0  visit the begginning of bank0
    // lane16 visit bank0 + 128 bytes
    unsigned sub_bank_idx           = lane_id / 16;
    unsigned sub_bank_elment_offset = sub_bank_idx * 32;
    unsigned s_idx                  = bank_element_offset + sub_bank_elment_offset;

    smem[s_idx] = sinf(static_cast<float>(lane_id));
    __syncthreads();

    if (tid < N) {
      data[tid] = smem[s_idx];
    }
  }

  __global__ void memory_padding() {
  }

} // namespace

auto main() noexcept(false) -> int {
  const int N            = 1 * 1'024 * 1'024;
  const std::size_t size = N * sizeof(float);
  float* buffer          = nullptr;
  cudaMalloc(&buffer, size);

  bank_conflict<<<1, 32>>>(buffer, N);
}
