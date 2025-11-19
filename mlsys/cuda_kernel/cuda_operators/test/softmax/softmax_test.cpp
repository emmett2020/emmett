#include <ATen/Context.h>
#include <c10/core/ScalarType.h>

#include "catch2_wrap.h"
#include "precision_proxy.h"
#include "softmax_impl.h"

TEST_CASE("Small shape", "[precision][softmax]") {
  torch::manual_seed(0);
  auto shape = GENERATE(std::vector<int>{8'000, 2, 3}, std::vector<int>{256, 1'024, 2'048});
  auto dtype = GENERATE(torch::kFloat);

  auto softmax       = Softmax{};
  softmax.input_size = shape;
  softmax.input_opt  = dtype;

  auto precision_proxy = MakePrecisionProxy(softmax);
  TestPrecision(precision_proxy);
}

