#include "rms_norm/rms_norm.h"

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
    unsigned int N,
    unsigned int C,
    unsigned int H,
    unsigned int W,
    float epsilon,
    float* __restrict__ output_ptr) {
    __shared__ float s_rms;
    __shared__ float s_ssum[32]; // square sum

    const unsigned num_elements = C * H * W;
    const unsigned tid          = threadIdx.x;
    const unsigned n            = blockIdx.x;
    const unsigned batch_offset  = n * C * H * W;

    constexpr unsigned vectorized_dim = 4;
    unsigned num_elements_compressed  = num_elements / vectorized_dim;

    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + batch_offset);
    float4* output_ptr4      = reinterpret_cast<float4*>(output_ptr + batch_offset);

    // 1. Compute input sum & square sum
    float ssum = 0.F;
    for (unsigned i = tid; i < num_elements_compressed; i += blockDim.x) {
      float4 data = input_ptr4[i];

      ssum += data.x * data.x;
      ssum += data.y * data.y;
      ssum += data.z * data.z;
      ssum += data.w * data.w;
    }

    // 2. Reduce
    ssum = block_reduce_sum(ssum, s_ssum);

    // 3. Compute mean & rstd
    if (tid == 0) {
      s_rms = rsqrtf(ssum / num_elements + epsilon);
    }
    __syncthreads();
    float rms = s_rms;

    // 4. Compute normalization
    for (unsigned i = tid; i < num_elements_compressed; i += blockDim.x) {
      float4 data = input_ptr4[i];

      float norm0 = (data.x) * rms;
      float norm1 = (data.y) * rms;
      float norm2 = (data.z) * rms;
      float norm3 = (data.w) * rms;

      output_ptr4[i] = make_float4(norm0, norm1, norm2, norm3);
    }
  }

  cudaError_t launch_rms_norm(
    const float* input_ptr,
    unsigned N,
    unsigned C,
    unsigned H,
    unsigned W,
    float epsilon,
    float* output_ptr) {
    const int blk_size = 256;
    const int num_blks = N;
    layer_norm_kernel<<<num_blks, blk_size>>>(input_ptr, N, C, H, W, epsilon, output_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_rms_norm(const torch::Tensor& input, float epsilon) {
    TORCH_CHECK(input.is_cuda(), "Tensors must be on CUDA device");

    auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shapes of input must be 4");
    unsigned N = shape[0];
    unsigned C = shape[1];
    unsigned H = shape[2];
    unsigned W = shape[3];


    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_rms_norm(input_ptr, N, C, H, W, epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
