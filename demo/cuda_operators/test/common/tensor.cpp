#include "tensor.h"

#include <torch/torch.h>

#include <range/v3/all.hpp>

auto CopyTensors(const Tensors &tensors, const torch::TensorOptions &options) -> Tensors {
  return ranges::views::all(tensors)
       | ranges::views::transform([&](const torch::Tensor &tensor) noexcept {
           return tensor.to(options);
         })
       | ranges::to<Tensors>();
}
