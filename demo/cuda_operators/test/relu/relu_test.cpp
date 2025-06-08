#include <c10/core/ScalarType.h>

#include "catch2_wrap.h"
#include "precision_proxy.h"
#include "relu_impl.h"

TEST_CASE("Small shape", "[precision][relu][fp32]") {
  auto shape = GENERATE(
    std::vector<int>{1},
    std::vector<int>{1, 2},
    std::vector<int>{1, 2, 3},
    std::vector<int>{8'000, 2, 3},
    std::vector<int>{256, 1'024, 2'048});
  auto dtype = GENERATE(torch::kFloat, torch::kBFloat16);

  auto relu       = Relu{};
  relu.input_size = shape;
  relu.input_opt  = dtype;

  auto precision_proxy = MakePrecisionProxy(relu);
  TestPrecision(precision_proxy);
}

