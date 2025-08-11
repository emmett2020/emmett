#include <pybind11/pybind11.h>

// #include "add/add.h"
// #include "sigmoid/sigmoid.h"
// #include "layer_norm/layer_norm.h"
// #include "group_norm/group_norm.h"
// #include "batch_norm/batch_norm.h"
// #include "flash_attention/flash_attention.h"
// #include "softmax/softmax.h"
// #include "sorting/sorting.h"
#include "conv/conv.h"

namespace cuda_op { } // namespace cuda_op

PYBIND11_MODULE(cuda_op, m) {
  m.doc() = "High performance cuda kernels";
  m.def("conv2d", &cuda_op::torch_conv2d);
}

// PYBIND11_MODULE(cuda_op, m) {
//   m.doc() = "High performance cuda kernels";
//   m.def("add",
//         &cuda_op::torch_add,
//         "A function that add two tensors",
//         pybind11::arg("a"),
//         pybind11::arg("b"));
//   m.def("group_norm", &cuda_op::torch_group_norm);
//   m.def("batch_norm", &cuda_op::torch_batch_norm);
//   m.def("layer_norm", &cuda_op::torch_layer_norm);
//   m.def("layer_norm_nlp", &cuda_op::torch_layer_norm_nlp);
//   m.def("sigmoid", &cuda_op::torch_sigmoid);
//   m.def("softmax", &cuda_op::torch_softmax);
//   m.def("safe_softmax", &cuda_op::torch_safe_softmax);
//   m.def("flash_attention", &cuda_op::torch_flash_attn);
//   m.def("sort", &cuda_op::torch_sorting);
// }
