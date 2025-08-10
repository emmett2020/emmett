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

  // Per N per ThreadBlock
  // CxHxW must be divisible by 4.
  __global__ void layer_norm_kernel(
    const float* __restrict__ input_ptr,
    const float* __restrict__ gamma_ptr,
    const float* __restrict__ beta_ptr,
    unsigned int N,
    unsigned int C,
    unsigned int H,
    unsigned int W,
    float epsilon,
    float* __restrict__ output_ptr) {
    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_sum[32];
    __shared__ float s_ssum[32]; // square sum

    const unsigned num_elements = C * H * W;
    const unsigned tid          = threadIdx.x;
    const unsigned n            = blockIdx.x;
    const unsigned batch_offst  = n * C * H * W;

    constexpr unsigned vectorized_dim = 4;
    unsigned num_elements_compressed  = num_elements / vectorized_dim;

    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + batch_offst);
    float4* output_ptr4      = reinterpret_cast<float4*>(output_ptr + batch_offst);

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

  // Input must be NHWC
  // gridDim.x == N * H * W
  // Per C per thread block
  __global__ void layer_norm_nlp_kernel(
    const float* __restrict__ input_ptr,
    const float* __restrict__ gamma_ptr,
    const float* __restrict__ beta_ptr,
    unsigned int N,
    unsigned int H,
    unsigned int W,
    unsigned int C,
    float epsilon,
    float* __restrict__ output_ptr) {
    __shared__ float s_sum[32];
    __shared__ float s_ssum[32];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;

    const unsigned n  = bid / (H * W);
    const unsigned hw = bid % (H * W);
    const unsigned h  = hw / W;
    const unsigned w  = hw % W;

    const unsigned offset = n * H * W * C + h * W * C + w * C;

    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + offset);
    const float4* gamma_ptr4 = reinterpret_cast<const float4*>(gamma_ptr);
    const float4* beta_ptr4  = reinterpret_cast<const float4*>(beta_ptr);
    float4* output_ptr4      = reinterpret_cast<float4*>(output_ptr + offset);

    // 1. Compute input sum and square sum
    float sum  = 0.0F;
    float ssum = 0.0F; // square sum
    for (int i = tid; i < C / 4; i += blockDim.x) {
      float4 input = input_ptr4[i];

      sum  += input.x;
      ssum += input.x * input.x;

      sum  += input.y;
      ssum += input.y * input.y;

      sum  += input.z;
      ssum += input.z * input.z;

      sum  += input.w;
      ssum += input.w * input.w;
    }

    // Reduce sum and square sum in block
    sum  = block_reduce_sum(sum, s_sum);
    ssum = block_reduce_sum(ssum, s_ssum);
    if (tid == 0) {
      float mean     = sum / C;
      float variance = ssum / C - mean * mean;

      s_mean = mean;
      s_rstd = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    for (int i = tid; i < C / 4; i += blockDim.x) {
      float4 input = input_ptr4[i];
      float4 gamma = gamma_ptr4[i];
      float4 beta  = beta_ptr4[i];

      float norm0 = (input.x - mean) * rstd * gamma.x + beta.x;
      float norm1 = (input.y - mean) * rstd * gamma.y + beta.y;
      float norm2 = (input.z - mean) * rstd * gamma.z + beta.z;
      float norm3 = (input.w - mean) * rstd * gamma.w + beta.w;

      output_ptr4[i] = make_float4(norm0, norm1, norm2, norm3);
    }
  }

  cudaError_t launch_layer_norm(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    unsigned N,
    unsigned C,
    unsigned H,
    unsigned W,
    float epsilon,
    float* output_ptr) {
    const int blk_size = 256;
    const int num_blks = N;
    layer_norm_kernel<<<num_blks, blk_size>>>(
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

  cudaError_t launch_layer_norm_nlp(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    unsigned N,
    unsigned H,
    unsigned W,
    unsigned C,
    float epsilon,
    float* output_ptr) {
    const int num_blks = N * H * W;
    const int blk_size = 128;
    layer_norm_nlp_kernel<<<num_blks, blk_size>>>(
      input_ptr,
      gamma_ptr,
      beta_ptr,
      N,
      H,
      W,
      C,
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
    TORCH_CHECK(shape.size() == 4, "shapes of input must be 4");
    unsigned N = shape[0];
    unsigned C = shape[1];
    unsigned H = shape[2];
    unsigned W = shape[3];

    unsigned num_elements = C * H * W;
    TORCH_CHECK(num_elements % 4 == 0, "HxW must be divisible by 4");

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr  = beta.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_layer_norm(input_ptr, gamma_ptr, beta_ptr, N, C, H, W, epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

  torch::Tensor torch_layer_norm_nlp(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon) {
    TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "Tensors must be on CUDA device");

    auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shapes of input must be 4");
    unsigned N = shape[0];
    unsigned H = shape[1];
    unsigned W = shape[2];
    unsigned C = shape[3];

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr  = beta.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(
      launch_layer_norm_nlp(input_ptr, gamma_ptr, beta_ptr, N, H, W, C, epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
