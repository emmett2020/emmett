#pragma once

#include <pybind11/pybind11.h>

namespace cuda_op {
  void add_op_add(pybind11::module& m);
}

