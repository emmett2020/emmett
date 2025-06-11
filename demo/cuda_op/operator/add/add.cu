#include "add.h"

#include <cstddef>
#include <cstring>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace cuda_op {
  template <class DataType>
  __global__ void
  add(const DataType* __restrict__ x1_buf,
      const DataType* __restrict__ x2_buf,
      int N,
      DataType* __restrict__ y_buf) {
    const auto stride = gridDim.x * blockDim.x;
    for (std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x; i < N; i += stride) {
      DataType x1 = x1_buf[i];
      DataType x2 = x2_buf[i];
      DataType y  = x1 + x2;
      y_buf[i]    = y;
    }
  }

  void launch_add_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // 检查张量是否在 GPU 上
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.device().is_cuda(), "c must be a CUDA tensor");

    // 检查张量数据类型
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(c.scalar_type() == torch::kFloat32, "c must be float32");

    // 检查张量形状
    int n = a.numel();
    TORCH_CHECK(b.numel() == n, "a and b must have same number of elements");
    TORCH_CHECK(c.numel() == n, "c must have same number of elements as a and b");

    // 获取数据指针
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr       = c.data_ptr<float>();

    // 设置 CUDA 内核参数
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    // 启动 CUDA 内核

    // 检查内核执行是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
  }

  template <class DataType>
  void launch_add(const DataType* x1_cuda, const DataType* x2_cuda, int N, DataType* y_cuda) {
    const int threads_per_blk = 256;
    const int num_blk         = 828;
    add<<<num_blk, threads_per_blk>>>(x1_cuda, x2_cuda, N, y_cuda);
  }

  // TODO: bind pybind11 type to launch_relu

  void add_op_add(pybind11::module& m) {
    m.def("add", &launch_add<float>, "A function that add two tensors");
  }

} // namespace cuda_op


