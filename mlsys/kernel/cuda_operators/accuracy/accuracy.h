#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace compare {
  struct Config {
    int top_k   = 5;
    double rtol = 1e-5;
    double atol = 1e-8;

    std::size_t chunk_size = 1'048'576; // 1MB chunks
  };

  struct Result {
    explicit operator bool() const {
      return is_close;
    }

    bool is_close             = false;
    double comparison_time_ms = -1;
    int mismatch_count        = 0;

    struct {
      double max;
      double mean;
      double median;
      double p95;
      double p99;
      double std_dev;
    } stats{};

    struct {
      std::vector<double> histogram;
      std::vector<double> edges;
      size_t bucket_count = 0;
    } distribution;

    struct DiffItem {
      double value;
      std::vector<int64_t> position;
      float golden;
      float actual;
    };

    std::vector<DiffItem> top_diffs;

    std::vector<int64_t> shapes;
    torch::TensorOptions options;
    std::string device_type;
    Config cfg;
  };

  Result AllClose(const torch::Tensor& golden, const torch::Tensor& actual, const Config& cfg = {});

  void PrintResult(const Result& result);
} // namespace compare

