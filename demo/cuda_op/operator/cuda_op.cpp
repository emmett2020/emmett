#include <pybind11/pybind11.h>

#include "add/add.h"
#include "sigmoid/sigmoid.h"
#include "layernorm/layernorm.h"
#include "groupnorm/groupnorm.h"
#include "batchnorm/batchnorm.h"

namespace cuda_op { } // namespace cuda_op

PYBIND11_MODULE(cuda_op, m) {
  m.doc() = "high performance cuda kernels";
  m.def("add",
        &cuda_op::torch_add,
        "A function that add two tensors",
        pybind11::arg("a"),
        pybind11::arg("b"));
  m.def("sigmoid", &cuda_op::torch_sigmoid);
  m.def("group_norm", &cuda_op::torch_group_norm);
  m.def("batch_norm", &cuda_op::torch_batch_norm);
  m.def("layer_norm", &cuda_op::torch_layer_norm);
}
