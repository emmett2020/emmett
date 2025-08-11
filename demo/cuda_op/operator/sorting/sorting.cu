#include "sorting/sorting.h"

#include <ATen/ops/copy.h>
#include <torch/extension.h>

#include "common/utils.h"

namespace cuda_op {

  __inline__ __device__ void swap(float& a, float& b, bool is_ascending) {
    if (a > b == is_ascending) {
      float tmp = a;
      a         = b;
      b         = tmp;
    }
  }

  // Say N = 16,
  // Then:
  // k = 2, 4, 8, 16
  // when k = 2,  j = 1
  // when k = 4,  j = 2, 1
  // when k = 8,  j = 4, 2, 1
  // when k = 16, j = 8, 4, 2, 1

  // When k = 2, j = 1,
  //    when i = 0, i^j = 1

  // When k = 4,
  //    when j = 2,
  //      when i = 0, i^j = 2,
  //      when i = 1, i^j = 3,
  //      when i = 2, i^j = 0,
  //      when i = 3, i^j = 1,
  //    when j = 1,
  //      when i = 0, i^j = 1,
  //      when i = 1, i^j = 0,

  // Each function call, we use num_threads=N to do work. They are divided into
  // small groups, each group contains num_threads=2*k, and the first k of each
  // group will handle ascending while the left part handles descending.
  __global__ void bitonic_sort(float* data, unsigned N, unsigned k, unsigned j) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
      return;
    }

    unsigned ixj = i ^ j;
    if (i < ixj && ixj < N) {
      if ((i & k) == 0) {
        swap(data[i], data[ixj], true);
      } else {
        swap(data[i], data[ixj], false);
      }
    }
  }

  cudaError_t launch_sort(float* data, int N, unsigned k, unsigned j) {
    constexpr int blk_size = 128;
    const int num_blks     = (N + blk_size - 1) / blk_size;
    bitonic_sort<<<num_blks, blk_size>>>(data, N, k, j);
    return cudaGetLastError();
  }

  __host__ __device__ int next_power_of_two(int n) {
    int pow2 = 1;
    while (pow2 < n)
      pow2 <<= 1;
    return pow2;
  }

  torch::Tensor torch_sorting(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Tensors must be on CUDA device");

    const int N         = input.numel();
    const int padding_N = next_power_of_two(N);
    TORCH_CHECK(N == padding_N, "N must be power of 2");

    auto output = input.clone();

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr      = output.data_ptr<float>();
    for (unsigned k = 2; k <= N; k <<= 1) {
      for (unsigned j = k >> 1; j > 0; j >>= 1) {
        launch_sort(output_ptr, N, k, j);
        cuda_check(cudaDeviceSynchronize());
      }
    }
    cuda_check(cudaDeviceSynchronize());
    return output;
  }


} // namespace cuda_op
