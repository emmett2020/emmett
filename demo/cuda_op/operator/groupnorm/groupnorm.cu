#include "groupnorm/groupnorm.h"

#include "common/utils.h"
#include <cstddef>
#include <cstring>

#include <torch/extension.h>

namespace cuda_op {
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
    float* stats_ptr   = reduce_ptr + 2 * blockDim.x;
    float* mean_ptr    = &stats_ptr[0];
    float* inv_std_ptr = &stats_ptr[1];

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

      float variance = (total_square - total_sum * mean) / group_size;
      if (variance < 0.f) {
        variance = 0.f;
      }
      variance     += epsilon;
      *mean_ptr     = mean;
      *inv_std_ptr  = rsqrtf(variance);
    }
    __syncthreads();

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

  template <class T>
  cudaError_t launch_group_norm(
    const T* input,
    const T* gamma,
    const T* beta,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float epsilon,
    T* y_cuda) {
    const int total_groups = N * num_groups;
    dim3 grid(total_groups);
    dim3 block(256);
    const size_t shared_size = (2 * block.x + 2) * sizeof(float);

    group_norm<<<grid, block, shared_size>>>(
      input,
      gamma,
      beta,
      N,
      C,
      H,
      W,
      num_groups,
      epsilon,
      y_cuda);
    return cudaGetLastError();
  }

  torch::Tensor torch_group_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int num_groups,
    float epsilon) {
    TORCH_CHECK(input.sizes().size() == 4, "");
    const int N = input.sizes()[0];
    const int C = input.sizes()[1];
    const int H = input.sizes()[2];
    const int W = input.sizes()[3];
    TORCH_CHECK(C % num_groups == 0, "num_groups must be divied by C");

    auto output = torch::zero(input);
    cuda_check(launch_group_norm(
      input.data_ptr<float>(),
      gamma.data_ptr<float>(),
      beta.data_ptr<float>(),
      N,
      C,
      H,
      W,
      num_groups,
      epsilon,
      output.data_ptr<float>()));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
