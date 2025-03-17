#pragma once

#include <vector>

#include <torch/types.h>

using Tensor  = torch::Tensor;
using Tensors = std::vector<Tensor>;

/// Return copied tensors by the given tensors. You can pass in options to
/// modify copy operation.
auto CopyTensors(const Tensors &tensors, const torch::TensorOptions &options) -> Tensors;

/// TODO: Convert data type description to type

