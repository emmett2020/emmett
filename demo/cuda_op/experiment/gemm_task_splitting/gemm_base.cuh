#pragma once
#include <stdexcept>
#include <string_view>

namespace {
  void throw_if(bool cond, std::string_view msg) {
    if (cond) {
      throw std::runtime_error{msg.data()};
    }
  }
} // namespace
