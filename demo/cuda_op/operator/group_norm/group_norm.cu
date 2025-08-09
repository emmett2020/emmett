#include "group_norm/group_norm.h"

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
    int G,
    float epsilon,
    T* output) {
    const int D            = C / G;
    const int num_elements = D * H * W;
    const int n            = blockIdx.x / G;
    const int g            = blockIdx.x % G;
    const int tid          = threadIdx.x;
    const int input_start  = n * C * H * W + g * D * H * W;

    __shared__ float s_sum[512];
    __shared__ float s_square_sum[512];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    /// 1. Calculate mean and rstd
    float sum        = 0;
    float sum_square = 0;
    for (int i = tid; i < num_elements; i += blockDim.x) {
      float input  = input_ptr[input_start + i];
      sum         += input;
      sum_square  += input * input;
    }
    s_sum[tid]        = sum;
    s_square_sum[tid] = sum_square;
    __syncthreads();

    // TODO: reduce in warp

    // reduce within thread block
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (tid < s) {
        s_sum[tid]        += s_sum[tid + s];
        s_square_sum[tid] += s_square_sum[tid + s];
      }
      __syncthreads();
    }

    // mean & rstd
    if (tid == 0) {
      float total_sum         = s_sum[0];
      float total_sum_square  = s_square_sum[0];
      float mean              = total_sum / num_elements;
      float variance          = (total_sum_square - total_sum * mean) / num_elements;
      variance                = variance > 0.0F ? variance : 0.0F;
      variance               += epsilon;
      s_mean                  = mean;
      s_rstd                  = rsqrtf(variance);
    }
    __syncthreads();

    // final
    float mean = s_mean;
    float rstd = s_rstd;
    for (int i = tid; i < num_elements; i += blockDim.x) {
      int pos          = input_start + i;
      int c            = g * D + i / (H * W);
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
    int G,
    float epsilon,
    T* output) {
    const int blk_size = 128;
    const int num_blks = N * G;
    group_norm<<<num_blks, blk_size>>>(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      N,
      C,
      H,
      W,
      G,
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
