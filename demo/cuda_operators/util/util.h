#pragma once

#include <ios>
#include <string_view>
#include <vector>
#include <span>
#include <cstdint>

auto convert_to_int64_vec(const std::vector<int>& int_vec) -> std::vector<int64_t>;


auto write_data_to_file(std::span<char> data,
                        std::string_view file_path,
                        std::ios_base::openmode mode = std::ios::binary) -> void;
