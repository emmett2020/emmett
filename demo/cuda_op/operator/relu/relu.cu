#include <cstddef>
#include <cstring>

#include <pybind11/pybind11.h>

namespace {
  template <class DataType>
  __global__ void relu(const DataType* __restrict__ x_buf, int N, DataType* __restrict__ y_buf) {
    const auto stride = gridDim.x * blockDim.x;
    for (std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x; i < N; i += stride) {
      DataType val = x_buf[i];
      y_buf[i]     = val > static_cast<DataType>(0) ? val : static_cast<DataType>(0);
    }
  }

  template <class DataType>
  void launch_relu(const DataType* x_cuda, int N, DataType* y_cuda) {
    const int threads_per_blk = 256;
    const int num_blk         = 828;
    relu<<<num_blk, threads_per_blk>>>(x_cuda, N, y_cuda);
  }
} // namespace
