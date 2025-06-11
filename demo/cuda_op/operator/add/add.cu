#include "add.h"

#include <cstddef>
#include <cstring>

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "common/utils.h"

namespace cuda_op {
  template <class T>
  __global__ void
  add(const T* __restrict__ a_cuda, const T* __restrict__ b_cuda, int N, T* __restrict__ c_cuda) {
    const auto stride = gridDim.x * blockDim.x;
    for (std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x; i < N; i += stride) {
      T a       = a_cuda[i];
      T b       = b_cuda[i];
      T c       = a + b;
      c_cuda[i] = c;
    }
  }

  template <class T>
  cudaError_t launch_add(const T* a_cuda, const T* b_cuda, int N, T* c_cuda) {
    const int threads_per_blk = 256;
    const int num_blk         = 828;
    add<<<num_blk, threads_per_blk>>>(a_cuda, b_cuda, N, c_cuda);
    return cudaGetLastError();
  }

  torch::Tensor add_wrap(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(),
                "Both tensors must be on CUDA device");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Input tensors must have same scalar type");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");

    auto c = torch::empty_like(a);

    if (a.scalar_type() == torch::kFloat32) {
      const float* a_ptr = a.data_ptr<float>();
      const float* b_ptr = b.data_ptr<float>();
      float* c_ptr       = c.data_ptr<float>();
      const int n        = a.numel();
      cuda_check(launch_add<float>(a_ptr, b_ptr, n, c_ptr));
    } else {
      TORCH_CHECK(false, "Unsupported data type ");
    }

    cuda_check(cudaDeviceSynchronize());
    return c;
  }

  void add_op_add(pybind11::module& m) {
    m.def("add",
          &add_wrap,
          "A function that add two tensors",
          pybind11::arg("a"),
          pybind11::arg("b"));
  }
} // namespace cuda_op

