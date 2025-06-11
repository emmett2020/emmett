#include <pybind11/pybind11.h>

#include "add/add.h"

namespace cuda_op { } // namespace cuda_op

PYBIND11_MODULE(cuda_op, m) {
  m.doc() = "high performance cuda kernels";
  cuda_op::add_op_add(m);
}
