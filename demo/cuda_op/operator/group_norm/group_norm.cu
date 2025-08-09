#include "group_norm/group_norm.h"

#include <torch/extension.h>

#include "common/utils.h"

namespace cuda_op {
  __device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int s = warpSize >> 1; s > 0; s >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
  }

  __device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int tid = threadIdx.x;
    const int lid = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    // Reduce in warp
    val = warp_reduce_sum(val);

    if (lid == 0) {
      shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = blockDim.x / warpSize;

    val = (tid < num_warps) ? shared[lid] : 0;

    if (wid == 0) {
      val = warp_reduce_sum(val);
    }

    // threadIdx.x == 0 contains final result.
    return val;
  }

  template <class T>
  __global__ void group_norm_interview(
    const T* __restrict__ input_ptr,
    const T* __restrict__ gamma_ptr,
    const T* __restrict__ beta_ptr,
    int N,
    int C,
    int H,
    int W,
    int G,
    float epsilon,
    T* __restrict__ output) {
    const int D            = C / G;
    const int num_elements = D * H * W;
    const int n            = blockIdx.x / G;
    const int g            = blockIdx.x % G;
    const int c_group      = g * D;
    const int tid          = threadIdx.x;
    const int input_start  = n * C * H * W + c_group * H * W;

    __shared__ float s_sum[32];
    __shared__ float s_square_sum[32];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    /// 1. Calculate mean and rstd within thread block.
    float sum        = 0;
    float square_sum = 0;
    for (int i = tid; i < num_elements; i += blockDim.x) {
      float input  = input_ptr[input_start + i];
      sum         += input;
      square_sum  += input * input;
    }

    sum        = block_reduce_sum(sum, s_sum);
    square_sum = block_reduce_sum(square_sum, s_square_sum);
    if (tid == 0) {
      float mean     = sum / num_elements;
      float variance = square_sum / num_elements - mean * mean;

      s_mean = mean;
      s_rstd = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // Load gamma and beta.
    __shared__ float s_gamma[32];
    __shared__ float s_beta[32];
    if (tid < D) {
      s_gamma[tid] = gamma_ptr[c_group + tid];
      s_beta[tid]  = beta_ptr[c_group + tid];
    }
    __syncthreads();

    const int hw = H * W;
    for (int i = tid; i < num_elements; i += blockDim.x) {
      int idx     = input_start + i;
      float input = input_ptr[idx];

      float normalized = (input - mean) * rstd;
      int c_in_group   = i / hw;
      normalized       = normalized * s_gamma[c_in_group] + s_beta[c_in_group];

      output[idx] = normalized;
    }
  }

  template <class T>
  __global__ void group_norm(
    const T* __restrict__ input_ptr,
    const T* __restrict__ gamma_ptr,
    const T* __restrict__ beta_ptr,
    int N,
    int C,
    int H,
    int W,
    int G,
    float epsilon,
    T* __restrict__ output) {
    const int D            = C / G;
    const int num_elements = D * H * W;
    const int n            = blockIdx.x / G;
    const int g            = blockIdx.x % G;
    const int c_group      = g * D;
    const int tid          = threadIdx.x;
    const int input_start  = n * C * H * W + c_group * H * W;

    __shared__ float s_sum[32];
    __shared__ float s_square_sum[32];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    /// 1. Calculate mean and rstd within thread block.
    float sum        = 0;
    float square_sum = 0;
    const int stride = blockDim.x;
    for (int i = tid; i < num_elements / 4; i += stride) {
      const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + input_start);

      float4 input  = input_ptr4[i];
      sum          += input.x;
      square_sum   += input.x * input.x;

      sum        += input.y;
      square_sum += input.y * input.y;

      sum        += input.z;
      square_sum += input.z * input.z;

      sum        += input.w;
      square_sum += input.w * input.w;
    }

    sum        = block_reduce_sum(sum, s_sum);
    square_sum = block_reduce_sum(square_sum, s_square_sum);
    if (tid == 0) {
      float mean     = sum / num_elements;
      float variance = square_sum / num_elements - mean * mean;

      s_mean = mean;
      s_rstd = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // Load gamma and beta.
    __shared__ float s_gamma[32];
    __shared__ float s_beta[32];
    if (tid < D) {
      s_gamma[tid] = gamma_ptr[c_group + tid];
      s_beta[tid]  = beta_ptr[c_group + tid];
    }
    __syncthreads();

    const int hw = H * W;
    for (int i = tid; i < num_elements; i += blockDim.x) {
      int idx     = input_start + i;
      float input = input_ptr[idx];

      float normalized = (input - mean) * rstd;
      int c_in_group   = i / hw;
      normalized       = normalized * s_gamma[c_in_group] + s_beta[c_in_group];

      output[idx] = normalized;
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
    const int blk_size = 256;
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
