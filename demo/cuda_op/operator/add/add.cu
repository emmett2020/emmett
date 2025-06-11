#include "add.h"

#include <cstddef>
#include <cstring>

#include <pybind11/pybind11.h>

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


