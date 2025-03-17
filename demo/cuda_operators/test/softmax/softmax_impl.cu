#include "softmax_impl.h"

#include <ATen/ops/sum.h>
#include <c10/core/ScalarType.h>

#include "util.h"
#include "softmax/softmax.cuh"
#include "accuracy.h"

Tensors Softmax::CreateInputs() const noexcept {
  auto size = convert_to_int64_vec(input_size);
  return {torch::randn(size, input_opt)};
}

Tensors Softmax::ComputeCpuGolden(const Tensors &inputs) {
  assert(inputs.size() == 1);
  auto shape          = inputs[0].sizes();
  auto flatten_tensor = inputs[0].view(-1);
  auto result         = torch::softmax(flatten_tensor, 0);
  return {result.view(shape)};
}

Tensors Softmax::ComputeActual(const Tensors &inputs) {
  assert(inputs.size() == 1);
  const auto &input      = inputs[0];
  const auto element_cnt = input.numel();
  auto output            = torch::zeros_like(input);

  if (input.dtype() == torch::kFloat) {
    launch_softmax<float>(static_cast<float *>(input.data_ptr()),
                          element_cnt,
                          static_cast<float *>(output.data_ptr()));
  } else if (input.dtype() == torch::kBFloat16) {
    launch_softmax<nv_bfloat16>(
      static_cast<nv_bfloat16 *>(input.data_ptr()),
      element_cnt,
      static_cast<nv_bfloat16 *>(output.data_ptr()));
  } else {
    assert(false && "unsupported");
  }

  return {output};
}

bool Softmax::Compare(const Tensor &golden, const Tensor &actual) const {
  std::cout << "Compare: \n";
  std::cout << "Golden: " << golden << "\n\n\n";
  std::cout << "Actual: " << actual << "\n\n\n";

  auto actual_sum = torch::sum(actual).item<float>();
  // assert(actual_sum == 1.0);

  auto [atol, rtol] = std::tuple<float, float>{0, 0};
  if (golden.dtype() == torch::kBFloat16) {
    atol = 0.0016;
    rtol = 0.0016;
  } else {
    atol = 0.0016;
    rtol = 0.0016;
  }

  auto config = compare::Config{};
  config.atol = atol;
  config.rtol = rtol;

  auto result = compare::AllClose(golden, actual, config);
  compare::PrintResult(result);
  assert(result);
  return result.is_close;
}

