#include "layer_norm/layer_norm.h"

#include <torch/extension.h>
#include "common/utils.h"

namespace cuda_op {

  __forceinline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int s = warpSize >> 1; s > 0; s >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
  }

  __forceinline__ __device__ float block_reduce_sum(float val, float* shared) {
    const unsigned num_warps = blockDim.x / warpSize;

    const unsigned tid = threadIdx.x;
    const unsigned lid = tid % warpSize;
    const unsigned wid = tid / warpSize;

    val = warp_reduce_sum(val);
    if (lid == 0) {
      shared[wid] = val;
    }
    __syncthreads();

    val = tid < num_warps ? shared[lid] : 0.F;
    if (wid == 0) {
      val = warp_reduce_sum(val);
    }

    return val;
  }

  // Per C per ThreadBlock
  __global__ void batch_norm_kernel(
    const float* __restrict__ input_ptr,
    const float* __restrict__ gamma_ptr,
    const float* __restrict__ beta_ptr,
    unsigned int N,
    unsigned int C,
    unsigned int H,
    unsigned int W,
    float epsilon,
    float* __restrict__ output_ptr) {
    __shared__ float s_sum[32];
    __shared__ float s_ssum[32];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;

    const unsigned c            = bid;
    const unsigned num_elements = N * H * W;

    // 1. Compute input sum and square sum
    float sum  = 0.0F;
    float ssum = 0.0F; // square sum
    for (int i = tid; i < num_elements; i += blockDim.x) {
      const unsigned n  = i / (H * W);
      const unsigned hw = i % (H * W);
      const unsigned h  = hw / W;
      const unsigned w  = hw % W;

      const unsigned offset = n * C * H * W + c * H * W + h * W + w;

      float input = input_ptr[offset];

      sum  += input;
      ssum += input * input;
    }

    // Reduce sum and square sum in block
    sum  = block_reduce_sum(sum, s_sum);
    ssum = block_reduce_sum(ssum, s_ssum);
    if (tid == 0) {
      float mean     = sum / num_elements;
      float variance = ssum / num_elements - mean * mean;

      s_mean = mean;
      s_rstd = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    for (int i = tid; i < num_elements; i += blockDim.x) {
      const unsigned n  = i / (H * W);
      const unsigned hw = i % (H * W);
      const unsigned h  = hw / W;
      const unsigned w  = hw % W;

      const unsigned offset = n * C * H * W + c * H * W + h * W + w;

      float input = input_ptr[offset];
      float gamma = gamma_ptr[c];
      float beta  = beta_ptr[c];

      float norm         = (input - mean) * rstd * gamma + beta;
      output_ptr[offset] = norm;
    }
  }

  cudaError_t launch_batch_norm(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    unsigned N,
    unsigned C,
    unsigned H,
    unsigned W,
    float epsilon,
    float* output_ptr) {
    const int num_blks = C;
    const int blk_size = 128;
    batch_norm_kernel<<<num_blks, blk_size>>>(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      N,
      C,
      H,
      W,
      epsilon,
      output_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon) {
    TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "Tensors must be on CUDA device");

    auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shapes of input must be 4");
    unsigned N = shape[0];
    unsigned C = shape[1];
    unsigned H = shape[2];
    unsigned W = shape[3];

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr  = beta.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_batch_norm(input_ptr, gamma_ptr, beta_ptr, N, C, H, W, epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
