#include "relu_impl.h"

#include <c10/core/ScalarType.h>

#include "util.h"
#include "relu/relu.cuh"

Tensors Relu::CreateInputs() const noexcept {
  auto size = convert_to_int64_vec(input_size);
  return {torch::randn(size, input_opt)};
}

Tensors Relu::ComputeCpuGolden(const Tensors &inputs) {
  assert(inputs.size() == 1);
  return {torch::relu(inputs[0])};
}

Tensors Relu::ComputeActual(const Tensors &inputs) {
  assert(inputs.size() == 1);
  const auto &input      = inputs[0];
  const auto element_cnt = input.numel();
  auto output            = torch::zeros_like(input);

  if (input.dtype() == torch::kFloat) {
    launch_relu<float>(static_cast<float *>(input.data_ptr()),
                       element_cnt,
                       static_cast<float *>(output.data_ptr()));
  } else if (input.dtype() == torch::kBFloat16) {
    launch_relu<nv_bfloat16>(
      static_cast<nv_bfloat16 *>(input.data_ptr()),
      element_cnt,
      static_cast<nv_bfloat16 *>(output.data_ptr()));
  } else {
    assert(false && "unsupported");
  }

  return {output};
}

bool Relu::Compare(const Tensor &golden, const Tensor &actual) const {
  if (!torch::allclose(golden, actual, 0, 0)) {
    return false;
  }
  return true;
}

