#include "util.h"

#include <fstream>
#include <ios>
#include <span>

#include <range/v3/all.hpp>

auto convert_to_int64_vec(const std::vector<int>& int_vec) -> std::vector<int64_t> {
  auto size = int_vec
            | ranges::views::transform([](int n) -> int64_t { return n; })
            | ranges::to<std::vector>();
  return size;
}

auto write_data_to_file(std::span<char> data,
                        std::string_view file_path,
                        std::ios_base::openmode mode) -> void {
  auto ofs = std::ofstream{};
  ofs.open(file_path.data(), mode);
  assert(ofs);
  ofs.write(reinterpret_cast<char*>(data.data()), static_cast<std::int64_t>(data.size()));
  ofs.close();
}

