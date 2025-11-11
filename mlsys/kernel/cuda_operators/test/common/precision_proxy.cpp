#include "precision_proxy.h"

#include <filesystem>
#include <fstream>

#include <range/v3/all.hpp>

#include "catch2_wrap.h"
#include "tensor.h"
#include "env_manager.h"

namespace {
  void DumpTensorToFile(const Tensor& tensor,
                        const std::string& file_name,
                        bool force_rewrite = true) {
    // TODO: Add customized path
    auto dumped_path = env::get("DEBUG_ARTIFACTS_DUMPED_PATH");
    if (dumped_path.empty()) {
      return;
    }
    dumped_path          += dumped_path.back() == '/' ? "debug" : "/debug";
    const auto file_path  = std::format("{}/{}", dumped_path, file_name);

    if (!force_rewrite) {
      if (std::filesystem::exists(file_path)) {
        throw std::runtime_error(std::format("file: {} already exists", file_path));
      }
    }

    if (!std::filesystem::exists(dumped_path)) {
      std::filesystem::create_directories(dumped_path);
    }

    auto file = std::ofstream(file_path, std::ios::out);
    if (!file) {
      throw std::runtime_error(std::format("can't open file: {}", file_path));
    }

    file << tensor << std::endl;
  }
} // namespace

void TestPrecision(pro::proxy<PrecisionProxy> op) {
  auto inputs_cpu    = op->CreateInputs();
  auto is_cpu_tensor = [](const Tensor& tensor) {
    return tensor.device().is_cpu();
  };
  REQUIRE(ranges::all_of(inputs_cpu, is_cpu_tensor));
  for (const auto& [index, tensor]: ranges::views::enumerate(inputs_cpu)) {
    DumpTensorToFile(tensor, std::format("input_tensor_{}", index));
  }

  auto goldens = op->ComputeCpuGolden(inputs_cpu);
  REQUIRE(ranges::all_of(goldens, is_cpu_tensor));
  for (const auto& [index, tensor]: ranges::views::enumerate(goldens)) {
    DumpTensorToFile(tensor, std::format("output_golden_tensor_{}", index));
  }

  auto inputs_cuda = CopyTensors(inputs_cpu, torch::kCUDA);
  auto actual_cuda = op->ComputeActual(inputs_cuda);
  auto actual_cpu  = CopyTensors(actual_cuda, torch::kCPU);
  for (const auto& [index, tensor]: ranges::views::enumerate(actual_cpu)) {
    DumpTensorToFile(tensor, std::format("output_actual_tensor_{}", index));
  }

  for (const auto& [golden, actual]: ranges::views::zip(goldens, actual_cpu)) {
    REQUIRE(op->Compare(golden, actual));
  }
}
