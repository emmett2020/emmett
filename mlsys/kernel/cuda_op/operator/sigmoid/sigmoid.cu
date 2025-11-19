#include "sigmoid.h"

#include <torch/extension.h>
#include <torch/types.h>

#include "common/utils.h"

namespace cuda_op {
  template <typename T>
  __global__ void sigmoid(const T* __restrict__ input_ptr, int N, T* __restrict__ output_ptr) {
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
      auto x        = input_ptr[i];
      auto o        = 1 / (1 + exp(-x));
      output_ptr[i] = o;
    }
  }

  __global__ void
  sigmoid_vectorized(const float* __restrict__ input_ptr, int N, float* __restrict__ output_ptr) {
    const float4* input_ptr4 = reinterpret_cast<const float4*>(input_ptr);
    float4* output_ptr4      = reinterpret_cast< float4*>(output_ptr);

    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N / 4; i += stride) {
      float4 x       = input_ptr4[i];
      auto o0        = 1 / (1 + exp(-x.x));
      auto o1        = 1 / (1 + exp(-x.y));
      auto o2        = 1 / (1 + exp(-x.z));
      auto o3        = 1 / (1 + exp(-x.w));
      output_ptr4[i] = make_float4(o0, o1, o2, o3);
    }
  }

  cudaError_t launch_sigmoid_vectorized(const float* input, int N, float* output) {
    const int blk_size  = 256;
    const int grid_size = (N + blk_size - 1) / blk_size / 4;
    sigmoid_vectorized<<<grid_size, blk_size>>>(input, N, output);
    return cudaGetLastError();
  }

  template <typename T>
  cudaError_t launch_sigmoid(const T* input, int N, T* output) {
    const int blk_size  = 256;
    const int grid_size = (N + blk_size - 1) / blk_size;
    sigmoid<<<grid_size, blk_size>>>(input, N, output);
    return cudaGetLastError();
  }

  torch::Tensor torch_sigmoid(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input must on cuda device");
    auto output = torch::zero(input);
    const int N = input.numel();
    if (input.scalar_type() == torch::kFloat32) {
      auto* input_ptr  = input.data_ptr<float>();
      auto* output_ptr = output.data_ptr<float>();
      // cuda_check(launch_sigmoid<float>(input_ptr, N, output_ptr));
      cuda_check(launch_sigmoid_vectorized(input_ptr, N, output_ptr));
    } else {
      std::string type_str = torch::toString(input.scalar_type());
      TORCH_CHECK(false, "Unsupported data type of input ", type_str);
    }
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

} // namespace cuda_op

