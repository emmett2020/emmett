#include "softmax/softmax.h"

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

  // Reduce in W dimensions.
  __global__ void softmax(
    const float* __restrict__ input_ptr,
    unsigned N,
    unsigned H,
    unsigned W,
    unsigned C,
    float* __restrict__ output_ptr) {
    __shared__ float s_exp[32];
    __shared__ float s_exp_sum_buffer[32]; // exp sum
    __shared__ float s_exp_sum;            // exp sum

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;

    const unsigned n  = bid / (H * W);
    const unsigned hw = bid % (H * W);
    const unsigned h  = hw / W;
    const unsigned w  = hw % W;

    const unsigned offset = n * H * W * C + h * W * C + w * C;

    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr + offset);
    float4* output_ptr4      = reinterpret_cast<float4*>(output_ptr + offset);

    float exp_sum = 0.0F; // square sum
    for (int i = tid; i < C / 4; i += blockDim.x) {
      float4 input = input_ptr4[i];
      float exp_x  = expf(input.x);
      float exp_y  = expf(input.y);
      float exp_z  = expf(input.z);
      float exp_w  = expf(input.w);

      exp_sum += exp_x + exp_y + exp_y + exp_w;
    }

    // Reduce sum and square sum in block
    exp_sum = block_reduce_sum(exp_sum, s_exp_sum_buffer);
    if (tid == 0) {
      s_exp_sum = exp_sum;
    }
    __syncthreads();
    exp_sum = s_exp_sum;

    for (int i = tid; i < C / 4; i += blockDim.x) {
      float4 input = input_ptr4[i];
      float exp_x  = expf(input.x) / exp_sum;
      float exp_y  = expf(input.y) / exp_sum;
      float exp_z  = expf(input.z) / exp_sum;
      float exp_w  = expf(input.w) / exp_sum;

      output_ptr4[i] = make_float4(exp_x, exp_y, exp_z, exp_w);
    }
  }

  cudaError_t
  launch_softmax_nhwc(const float* input_ptr, int N, int H, int W, int C, float* output) {
    const int blk_size = 128;
    const int num_blks = N * H * W;
    softmax<<<num_blks, blk_size>>>(input_ptr, N, H, W, C, output);
    return cudaGetLastError();
  }

  torch::Tensor torch_softmax(const torch::Tensor& input) {
    const auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shape size must be 4");
    const int N = shape[0];
    const int H = shape[1];
    const int W = shape[2];
    const int C = shape[3];

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_softmax_nhwc(input_ptr, N, H, W, C, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

} // namespace cuda_op
