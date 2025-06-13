#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cstring>

namespace {
  template <class T>
  __global__ void group_norm(
    const T* __restrict__ input_ptr,
    const T* __restrict__ gamma_ptr,
    const T* __restrict__ beta_ptr,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float epsilon,
    T* __restrict__ output_ptr) {
    const int channels_per_group = C / num_groups;
    const int group_size         = channels_per_group * H * W;
    const int group_id           = blockIdx.x;
    const int n                  = group_id / num_groups;
    const int g                  = group_id % num_groups;
    const int input_start        = n * C * H * W + g * channels_per_group * H * W;

    extern __shared__ char shared_mem[];
    float* reduce_ptr  = reinterpret_cast<float*>(shared_mem);
    float* mean_ptr    = reinterpret_cast<float*>(shared_mem);
    float* inv_std_ptr = mean_ptr + 1;

    // Calculate
    float sum_input  = 0;
    float sum_square = 0;

    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
      T input       = input_ptr[i];
      output_ptr[i] = input > static_cast<T>(0) ? input : static_cast<T>(0);
    }
  }
} // namespace

template <class T>
void launch_group_norm(const T* x_cuda, int N, T* y_cuda) {
  const int threads_per_blk = 256;
  const int num_blk         = 828;
  group_norm<<<num_blk, threads_per_blk>>>(x_cuda, N, y_cuda);
}
