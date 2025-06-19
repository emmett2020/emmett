#include "layernorm/layernorm.h"

#include <torch/extension.h>
#include "common/utils.h"

namespace cuda_op {

  __inline__ __device__ float warp_reduce_sum(float val) {
    for (int i = 16; i > 0; i /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }
    return val;
  }

  __inline__ __device__ void atomic_add_block(float* addr, float val) {
    unsigned active = __activemask();
    unsigned mask   = __ballot_sync(active, threadIdx.x % warpSize == 0);
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(addr, val / __popc(mask));
    }
  }

  __global__ void layer_norm_kernel(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    int N,
    int C,
    float epsilon,
    float* output_ptr) {
    __shared__ float s_sum_input;
    __shared__ float s_sum_square;

    const int tid         = threadIdx.x;
    const int input_start = blockIdx.x * C;

    float sum_input = 0;
    for (int i = tid; i < C; i += blockDim.x) {
      float input  = input_ptr[input_start + i];
      sum_input   += input;
    }
    sum_input = warp_reduce_sum(sum_input);
    if (threadIdx.x % warpSize == 0) {
      atomic_add_block(&s_sum_input, sum_input);
    }
    __syncthreads();
    float mean = s_sum_input / C;
    __syncthreads();

    float sum_square = 0;
    for (int i = tid; i < C; i += blockDim.x) {
      float input  = input_ptr[input_start + i];
      sum_square  += (input - mean) * (input - mean);
    }
    sum_square = warp_reduce_sum(sum_square);
    if (threadIdx.x % warpSize == 0) {
      atomic_add_block(&s_sum_square, sum_square);
    }
    __syncthreads();
    float variance = s_sum_square / C;
    float inv_std  = rsqrtf(variance + epsilon);

    for (int i = tid; i < C; i += blockDim.x) {
      float input                 = input_ptr[input_start + i];
      float normalized            = (input - mean) * inv_std;
      normalized                  = normalized * gamma_ptr[i] + beta_ptr[i];
      output_ptr[input_start + i] = normalized;
    }
  }

  cudaError_t launch_layer_norm(
    const float* input_ptr,
    const float* gamma_ptr,
    const float* beta_ptr,
    int N,
    int C,
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
      epsilon,
      output_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_layer_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon) {
    auto output = torch::zero(input);
    auto shape  = input.sizes();
    TORCH_CHECK(shape.size() == 2, "shapes of input must be 2");
    TORCH_CHECK(input.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "Tensors must be on CUDA device");

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr  = beta.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(
      launch_layer_norm(input_ptr, gamma_ptr, beta_ptr, shape[0], shape[1], epsilon, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
