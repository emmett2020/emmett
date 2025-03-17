#include "accuracy.h"

#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <torch/types.h>
#include <vector>
#include <chrono>
#include <print>

#include <range/v3/all.hpp>
#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

namespace compare {
  using Tensor = torch::Tensor;

  namespace {
    [[nodiscard]] Tensor ComputeChunkDiff(
      const torch::Tensor& golden,
      const torch::Tensor& actual,
      int64_t start,
      int64_t end) {
      auto a_chunk = golden.narrow(0, start, end - start);
      auto b_chunk = actual.narrow(0, start, end - start);

      auto diff = torch::abs(a_chunk - b_chunk);
      return diff;
    }

    auto DtypeToStr(torch::TensorOptions options) -> std::string_view {
      if (options.dtype() == torch::kFloat32) {
        return "float";
      }
      if (options.dtype() == torch::kBFloat16) {
        return "bfloat16";
      }
      return "unknown";
    }

    // Function to unravel a flat index into multi-dimensional coordinates
    std::vector<int64_t> UnravelIndex(int64_t flat_idx, const at::IntArrayRef& shape) {
      auto coords    = std::vector<int64_t>(shape.size());
      auto remaining = flat_idx;

      // Iterate from last dimension to first (row-major order)
      for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        coords[i] = remaining % shape[i];
        remaining = remaining / shape[i];
      }
      return coords;
    }
  } // namespace

  namespace {
    class RtolComparator {
    private:
      Config cfg_;

    public:
      explicit RtolComparator(Config config)
        : cfg_(config) {
      }

      [[nodiscard]] Result Compare(const torch::Tensor& golden, const torch::Tensor& actual) const {
        auto result = Result{};
        auto start  = std::chrono::high_resolution_clock::now();

        result.options = golden.options();
        ranges::copy(golden.sizes().begin(),
                     golden.sizes().end(),
                     std::back_inserter(result.shapes));

        const int64_t num_elements = golden.numel();

        auto full_diff     = torch::abs(golden - actual);
        auto tolerance     = cfg_.atol + cfg_.rtol * torch::abs(actual);
        auto mismatch_mask = full_diff > tolerance;

        result.mismatch_count = torch::sum(mismatch_mask).item<int>();
        result.is_close       = (result.mismatch_count == 0);

        result.stats.max     = full_diff.max().item<double>();
        result.stats.mean    = full_diff.mean().item<double>();
        result.stats.std_dev = full_diff.std().item<double>();

        auto real_topk = std::min(cfg_.top_k, result.mismatch_count);

        if (real_topk > 0) {
          auto masked_diff       = full_diff.masked_fill(~mismatch_mask, 0.0);
          auto [values, indices] = torch::topk(masked_diff.flatten(), real_topk);
          result.top_diffs.reserve(real_topk);

          auto golden_flat = golden.flatten();
          auto actual_flat = actual.flatten();

          for (int i = 0; i < real_topk; ++i) {
            auto idx = indices[i].item<int64_t>();
            auto pos = UnravelIndex(idx, golden.sizes());

            result.top_diffs.push_back(
              {values[i].item<double>(),
               pos,
               golden_flat[idx].item<float>(),
               actual_flat[idx].item<float>()});
          }
        }

        // 性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        result.comparison_time_ms =
          std::chrono::duration<double, std::milli>(end_time - start).count();

        return result;
      }
    };
  } // namespace

  Result AllClose(const torch::Tensor& golden, const torch::Tensor& actual, const Config& cfg) {
    auto comparator = RtolComparator(cfg);
    return comparator.Compare(golden, actual);
  }

  void PrintResult(const Result& result) {
    auto str  = std::format("Result: {}\n", result.is_close);
    str      += std::format("Comparison time: {} ms\n", result.comparison_time_ms);
    str      += std::format("atol: {}, rtol: {}\n", result.cfg.atol, result.cfg.rtol);
    str      += std::format("Mismatch count: {}\n", result.mismatch_count);
    str      += std::format("Shape: ");
    for (auto shape: result.shapes) {
      str += std::format("{}, ", shape);
    }
    str += "\n";

    str += std::format("dtype: {}\n", DtypeToStr(result.options));
    for (const auto& top_diff: result.top_diffs) {
      str += std::format("golden: {}, actual: {}\n", top_diff.golden, top_diff.actual);
    }

    std::print("{}", str);
  }
} // namespace compare
