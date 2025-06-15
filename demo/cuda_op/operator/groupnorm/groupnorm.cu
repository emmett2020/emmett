#include "groupnorm/groupnorm.h"

#include <torch/extension.h>

#include "common/utils.h"

namespace cuda_op {
  //

  template <class T>
  __global__ void group_norm(
    const T* input_ptr,
    const T* gamma_ptr,
    const T* beta_ptr,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float epsilon,
    T* output) {
    const int channels_per_group = C / num_groups;
    const int group_size         = channels_per_group * H * W;
    const int group_idx          = blockIdx.x;
    const int n                  = group_idx / num_groups;
    const int g                  = group_idx % num_groups;
    const int tid                = threadIdx.x;
    const int input_start        = n * C * H * W + g * channels_per_group * H * W;

    extern __shared__ char shared_mem[];
    float* reduce_ptr = reinterpret_cast<float*>(shared_mem);
    float* stat_ptr   = reduce_ptr + 2 * blockDim.x;
    float* mean_ptr   = &stat_ptr[0];
    float* rstd_ptr   = &stat_ptr[1];

    float sum_input        = 0;
    float sum_input_square = 0;
    for (int i = tid; i < group_size; i += blockDim.x) {
      float input       = input_ptr[input_start + i];
      sum_input        += input;
      sum_input_square += input * input;
    }
    reduce_ptr[tid]              = sum_input;
    reduce_ptr[tid + blockDim.x] = sum_input_square;
    __syncthreads();

    // reduce within thread block
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (tid < s) {
        reduce_ptr[tid]              += reduce_ptr[tid + s];
        reduce_ptr[tid + blockDim.x] += reduce_ptr[tid + blockDim.x + s];
      }
      __syncthreads();
    }

    // mean & rstd
    if (tid == 0) {
      float total_sum         = reduce_ptr[0];
      float total_sum_square  = reduce_ptr[blockDim.x];
      float mean              = total_sum / group_size;
      float variance          = (total_sum_square - total_sum * mean) / group_size;
      variance                = std::max(variance, 0.0f);
      variance               += epsilon;
      *mean_ptr               = mean;
      *rstd_ptr               = rsqrtf(variance);
    }
    __syncthreads();

    // final
    float mean = *mean_ptr;
    float rstd = *rstd_ptr;
    for (int i = tid; i < group_size; i += blockDim.x) {
      int pos          = input_start + i;
      int c            = g * channels_per_group + i / (H * W);
      float input      = input_ptr[pos];
      float normalized = (input - mean) * rstd;
      normalized       = normalized * gamma_ptr[c] + beta_ptr[c];
      output[pos]      = normalized;
    }
  }

  template <class T>
  cudaError_t launch_group_norm(
    const T* input_ptr,
    const T* gamma_ptr,
    const T* beta_ptr,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float epsilon,
    T* output) {
    const int blk_size        = 256;
    const int num_blks        = N * num_groups;
    const int shared_mem_size = (2 * blk_size + 2) * sizeof(float);
    group_norm<<<num_blks, blk_size, shared_mem_size>>>(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      N,
      C,
      H,
      W,
      num_groups,
      epsilon,
      output);
    return cudaGetLastError();
  }

  torch::Tensor torch_group_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int num_groups,
    float epsilon) {
    const auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shape size must be 4");
    const int N = shape[0];
    const int C = shape[1];
    const int H = shape[2];
    const int W = shape[3];
    TORCH_CHECK(C % num_groups == 0, "num_groups must be divied by C");
    auto output       = torch::zero(input);
    float* input_ptr  = input.data_ptr<float>();
    float* gamma_ptr  = gamma.data_ptr<float>();
    float* beta_ptr   = beta.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    cuda_check(launch_group_norm(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      N,
      C,
      H,
      W,
      num_groups,
      epsilon,
      output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
