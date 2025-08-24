#include "softmax/softmax.h"

#include <cstdint>
#include <torch/extension.h>

#include "common/utils.h"

namespace cuda_op {
  /// WARN: Not fully validated. But we could refer this method.
  template <typename T, typename Op>
  __device__ __forceinline__ T warp_reduce(T val, Op op, unsigned mask = 0xffffffff) {
    constexpr int warp_size = 32;
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "warp_reduce only supports 4-byte or 8-byte types");

    union Converter {
      T orig;
      typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type bits;
    };

    Converter current;
    current.orig = val;

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
      Converter other;
      if constexpr (sizeof(T) == 4) {
        other.bits = __shfl_down_sync(mask, current.bits, offset);
      } else {
        other.bits = __shfl_down_sync(mask, current.bits, offset, warp_size);
      }
      current.orig = op(current.orig, other.orig);
    }

    // 广播结果到整个warp
    if constexpr (sizeof(T) == 4) {
      current.bits = __shfl_sync(mask, current.bits, 0);
    } else {
      current.bits = __shfl_sync(mask, current.bits, 0, warp_size);
    }

    return current.orig;
  }

  // Reduce in C dimensions.
  // Ln+1 = Ln * exp(max(N) - max(N+1)) + exp(Xn+1 - max(N+1))
  // Channel <= 128
  __global__ void safe_softmax(
    const float* __restrict__ input_ptr,
    unsigned N,
    unsigned H,
    unsigned W,
    unsigned C,
    float* __restrict__ output_ptr) {
    __shared__ float s_exp_sum_buffer[32]; // buffer for reduce exp sum
    __shared__ float s_exp_sum;            // exp sum

    __shared__ float row_sum[128];

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;

    const unsigned n  = bid / (H * W);
    const unsigned hw = bid % (H * W);
    const unsigned h  = hw / W;
    const unsigned w  = hw % W;

    const unsigned offset = n * H * W * C + h * W * C + w * C;

    float exp_sum = 0.F;
    for (int i = tid; i < C; i += blockDim.x) {
      float input  = input_ptr[offset + i];
      float exp_v  = expf(input);
      exp_sum     += exp_v;
    }

    // Reduce sum and square sum in block
    exp_sum = block_reduce_sum(exp_sum, s_exp_sum_buffer);
    if (tid == 0) {
      s_exp_sum = exp_sum;
    }
    __syncthreads();
    exp_sum = s_exp_sum;

    for (int i = tid; i < C; i += blockDim.x) {
      float input            = input_ptr[offset + i];
      float v                = expf(input) / exp_sum;
      output_ptr[offset + i] = v;
    }
  }

  cudaError_t
  launch_safe_softmax_nhwc(const float* input_ptr, int N, int H, int W, int C, float* output) {
    const int blk_size = 128;
    const int num_blks = N * H * W;
    safe_softmax<<<num_blks, blk_size>>>(input_ptr, N, H, W, C, output);
    return cudaGetLastError();
  }

  torch::Tensor torch_safe_softmax(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Tensors must be on CUDA device");

    const auto shape = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shape size must be 4");
    const int N = shape[0];
    const int H = shape[1];
    const int W = shape[2];
    const int C = shape[3];

    auto output = torch::zero(input);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    cuda_check(launch_safe_softmax_nhwc(input_ptr, N, H, W, C, output_ptr));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

} // namespace cuda_op
