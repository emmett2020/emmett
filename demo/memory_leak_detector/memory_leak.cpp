#include <iostream>

namespace {
  void malloc_without_free() {
    // NOLINTNEXTLINE
    auto *m = ::malloc(2'048);
  }
} // namespace

auto main() -> int {
  malloc_without_free();
  return 0;
}
