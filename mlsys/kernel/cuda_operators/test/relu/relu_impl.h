#pragma once

#include <ATen/core/ATen_fwd.h>
#include <c10/util/typeid.h>
#include <torch/torch.h>

#include "tensor.h"

struct Relu {
  Tensors CreateInputs() const noexcept;
  Tensors ComputeCpuGolden(const Tensors& input);
  Tensors ComputeActual(const Tensors& input);
  bool Compare(const Tensor& golden, const Tensor& actual) const;

  std::vector<int> input_size;
  torch::TensorOptions input_opt;
};


