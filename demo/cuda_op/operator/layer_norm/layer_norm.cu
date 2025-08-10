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
  // HxW must be divisible by 4.
  __global__ void layer_norm_kernel(
    const float* __restrict__ input_ptr,
    const float* __restrict__ gamma_ptr,
    const float* __restrict__ beta_ptr,
    unsigned int C,
    unsigned int H,
    unsigned int W,
    float epsilon,
    float* __restrict__ output_ptr) {
    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_sum[32];
    __shared__ float s_ssum[32]; // square sum

    const unsigned num_elements  = H * W;
    const unsigned tid           = threadIdx.x;
    const unsigned c             = blockIdx.x;
    const unsigned channel_offst = c * H * W;

    constexpr unsigned vectorized_dim = 4;
    unsigned num_elements_compressed  = num_elements / vectorized_dim;

    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + channel_offst);
    float4* output_ptr4      = reinterpret_cast<float4*>(output_ptr + channel_offst);

    // 1. Compute input sum & square sum
    float sum  = 0.F;
    float ssum = 0.F;
    for (unsigned i = tid; i < num_elements_compressed; i += blockDim.x) {
      float4 data = input_ptr4[i];

      sum  += data.x;
      ssum += data.x * data.x;
      sum  += data.y;
      ssum += data.y * data.y;
      sum  += data.z;
      ssum += data.z * data.z;
      sum  += data.w;
      ssum += data.w * data.w;
    }

    // 2. Reduce
    sum  = block_reduce_sum(sum, s_sum);
    ssum = block_reduce_sum(ssum, s_ssum);

    // 3. Compute mean & rstd
    if (tid == 0) {
      float mean     = sum / num_elements;
      float variance = ssum / num_elements - mean * mean;

      s_mean = mean;
      s_rstd = rsqrtf(variance + epsilon);
    }
    __syncthreads();
    float mean = s_mean;
    float rstd = s_rstd;

    // 4. Compute normalization
    for (unsigned i = tid; i < num_elements_compressed; i += blockDim.x) {
      float4 data = input_ptr4[i];

      const float4* gamma_ptr4 = reinterpret_cast<const float4*>(gamma_ptr);
      const float4* beta_ptr4  = reinterpret_cast<const float4*>(beta_ptr);
      float4 gamma             = gamma_ptr4[i];
      float4 beta              = beta_ptr4[i];

      float norm0 = (data.x - mean) * rstd * gamma.x + beta.x;
      float norm1 = (data.y - mean) * rstd * gamma.y + beta.y;
      float norm2 = (data.z - mean) * rstd * gamma.z + beta.z;
      float norm3 = (data.w - mean) * rstd * gamma.w + beta.w;

      output_ptr4[i] = make_float4(norm0, norm1, norm2, norm3);
    }
  }

  cudaError_t launch_layer_norm(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    int C,
    int H,
    int W,
    float epsilon,
    float* output_ptr) {
    const int blk_size = 256;
    const int num_blks = C;
    layer_norm_kernel<<<num_blks, blk_size>>>(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      C,
      H,
      W,
      epsilon,
      output_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon) {
    TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "Tensors must be on CUDA device");

    auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 3, "shapes of input must be 3");
    unsigned C = shape[0];
    unsigned H = shape[1];
    unsigned W = shape[2];

    unsigned num_elements = H * W;
    TORCH_CHECK(num_elements % 4 == 0, "HxW must be divisible by 4");

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr  = beta.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_layer_norm(input_ptr, gamma_ptr, beta_ptr, C, H, W, epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
