#include "sigmoid.h"

#include <torch/extension.h>
#include <torch/types.h>

#include "common/utils.h"

namespace cuda_op {
  template <typename T>
  __global__ void sigmoid(const T* input, int N, T* output) {
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; ++i) {
      auto v    = -1 * input[i];
      auto o    = 1 / (1 + exp(v));
      output[i] = o;
    }
  }

  template <typename T>
  cudaError_t launch_sigmoid(const T* input, int N, T* output) {
    const int blk_size  = 128;
    const int grid_size = (N + blk_size - 1) / blk_size * blk_size;
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
      cuda_check(launch_sigmoid<float>(input_ptr, N, output_ptr));
    } else {
      std::string type_str = torch::toString(input.scalar_type());
      TORCH_CHECK(false, "Unsupported data type of input ", type_str);
    }
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

} // namespace cuda_op

