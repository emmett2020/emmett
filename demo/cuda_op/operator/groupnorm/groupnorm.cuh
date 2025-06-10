#pragma once

#include <cstddef>
#include <cstring>

namespace {
  template <class T>
  __global__ void group_norm(const T* __restrict__ x_buf, int N, T* __restrict__ y_buf) {
    const auto stride = gridDim.x * blockDim.x;
    for (std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x; i < N; i += stride) {
      T val    = x_buf[i];
      y_buf[i] = val > static_cast<T>(0) ? val : static_cast<T>(0);
    }
  }
} // namespace

template <class T>
void launch_group_norm(const T* x_cuda, int N, T* y_cuda) {
  const int threads_per_blk = 256;
  const int num_blk         = 828;
  group_norm<<<num_blk, threads_per_blk>>>(x_cuda, N, y_cuda);
}
