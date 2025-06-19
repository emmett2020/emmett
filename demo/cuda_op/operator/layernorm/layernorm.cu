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
    unsigned mask = __activemask();
    __ballot_sync(mask, threadIdx.x % warpSize == 0);
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(addr, val / __popc(mask));
    }
  }

  __global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    int N,
    int C,
    float epsilon,
    float* output) {
    __shared__ float s_mean, s_var;
    const int tid = threadIdx.x;
    for (int i = tid; i < C; i += blockDim.x) {
    }
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
    cuda_check(launch_layer_norm(
      input_ptr,
      output_ptr,
      gamma_ptr,
      beta_ptr,
      shape[0],
      shape[1],
      shape[2],
      shape[3],
      epsilon));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
