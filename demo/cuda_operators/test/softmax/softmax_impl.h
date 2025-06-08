#pragma once

#include <vector>

#include "tensor.h"

struct Softmax {
  Tensors CreateInputs() const noexcept;
  Tensors ComputeCpuGolden(const Tensors& input);
  Tensors ComputeActual(const Tensors& input);
  bool Compare(const Tensor& golden, const Tensor& actual) const;

  std::vector<int> input_size;
  torch::TensorOptions input_opt;
};


