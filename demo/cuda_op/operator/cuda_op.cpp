#include <pybind11/pybind11.h>

#include "add/add.h"
#include "sigmoid/sigmoid.h"

namespace cuda_op { } // namespace cuda_op

PYBIND11_MODULE(cuda_op, m) {
  m.doc() = "high performance cuda kernels";
  m.def("add",
        &cuda_op::torch_add,
        "A function that add two tensors",
        pybind11::arg("a"),
        pybind11::arg("b"));
  m.def("sigmoid", &cuda_op::torch_sigmoid);
}
