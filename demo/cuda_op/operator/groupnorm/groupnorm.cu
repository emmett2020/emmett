#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_math.h>
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

    // Calculate sum_input and sum_square
    float sum_input  = 0;
    float sum_square = 0;
    const int tid    = threadIdx.x;
    for (int idx = tid; idx < group_size; idx += blockDim.x) {
      float input  = input_ptr[input_start + idx];
      sum_input   += input;
      sum_square  += input * input;
    }

    // Reduce in thread block
    reduce_ptr[tid]              = sum_input;
    reduce_ptr[tid + blockDim.x] = sum_square;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        reduce_ptr[tid]              += reduce_ptr[tid + s];
        reduce_ptr[tid + blockDim.x] += reduce_ptr[tid + blockDim.x + s];
      }
      __syncthreads();
    }

    // Calculate mean and variance
    if (tid == 0) {
      const float total_sum    = reduce_ptr[0];
      const float total_square = reduce_ptr[blockDim.x];
      const float mean         = total_sum / group_size;

      float variance = (total_square - mean * total_sum) / group_size;
      if (variance < 0.f) {
        variance = 0.f;
      }
      variance     += epsilon;
      *mean_ptr     = mean;
      *inv_std_ptr  = rsqrtf(variance);
    }

    const float mean       = *mean_ptr;
    const float inv_std    = *inv_std_ptr;
    const int spatial_size = H * W;
    for (int idx = tid; idx < group_size; idx += blockDim.x) {
      const int pos        = input_start + idx;
      const float val      = input_ptr[pos];
      float normalized     = (val - mean) * inv_std;
      const int c_in_group = idx / spatial_size;
      const int c          = g * channels_per_group + c_in_group;

      normalized      = normalized * gamma_ptr[c] + beta_ptr[c];
      output_ptr[pos] = normalized;
    }
  }
} // namespace

template <class T>
void launch_group_norm(const T* x_cuda, int N, T* y_cuda) {
  const int threads_per_blk = 256;
  const int num_blk         = 828;
  group_norm<<<num_blk, threads_per_blk>>>(x_cuda, N, y_cuda);
}
