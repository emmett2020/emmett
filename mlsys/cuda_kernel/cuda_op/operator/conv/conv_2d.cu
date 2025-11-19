#include "conv/conv.h"

#include <torch/extension.h>
#include "common/utils.h"

namespace cuda_op {
  // Per thread per output position.
  __global__ void conv2d_kernel(
    const float* input_ptr,
    const float* kernel_ptr,
    unsigned IH,
    unsigned IW,
    unsigned KH,
    unsigned KW,
    float* output_ptr) {
    const unsigned OH = IH - KH + 1;
    const unsigned OW = IW - KW + 1;
    const unsigned oh = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (oh >= OH || ow >= OW) {
      return;
    }
    float sum = 0.0F;
    for (int kh = 0; kh < KH; ++kh) {
      for (int kw = 0; kw < KW; ++kw) {
        int kernel_offset = kh * KW + kw;
        int input_offset  = (oh + kh) * IW + (ow + kw);

        sum += input_ptr[input_offset] * kernel_ptr[kernel_offset];
      }
    }
    output_ptr[oh * OW + ow] = sum;
  }

  cudaError_t launch_conv2d(
    const float* input_ptr,
    const float* kernel_ptr,
    unsigned IH,
    unsigned IW,
    unsigned KH,
    unsigned KW,
    float* output_ptr) {
    constexpr std::size_t block_size = 16;

    dim3 thread_dim(block_size, block_size);
    dim3 block_dim((IW - KW + 1 + block_size - 1) / block_size,
                   (IH - KH + 1 + block_size - 1) / block_size);
    conv2d_kernel<<<block_dim, thread_dim>>>(input_ptr, kernel_ptr, IH, IW, KH, KW, output_ptr);
    return cudaGetLastError();
  }

  torch::Tensor torch_conv2d(const torch::Tensor& input, const torch::Tensor& kernel) {
    TORCH_CHECK(input.is_cuda() && kernel.is_cuda(), "Tensors must be on CUDA device");

    auto input_shape = input.sizes();
    TORCH_CHECK(input_shape.size() == 4, "shapes of input must be 4");
    TORCH_CHECK(input_shape[0] == 1 && input_shape[1] == 1, "unsupported");
    unsigned IH = input_shape[2];
    unsigned IW = input_shape[3];

    auto kernel_shape = kernel.sizes();
    TORCH_CHECK(kernel_shape.size() == 4, "shapes of input must be 4");
    TORCH_CHECK(kernel_shape[0] == 1 && kernel_shape[1] == 1, "unsupported");
    unsigned KH = kernel_shape[2];
    unsigned KW = kernel_shape[3];

    unsigned OH = (IH - KH + 1);
    unsigned OW = (IW - KW + 1);

    auto output = torch::zeros({1, 1, OH, OW}, input.options());

    const float* input_ptr  = input.data_ptr<float>();
    const float* kernel_ptr = kernel.data_ptr<float>();
    float* output_ptr       = output.data_ptr<float>();
    cuda_check(launch_conv2d(input_ptr, kernel_ptr, IH, IW, KH, KW, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }
} // namespace cuda_op
