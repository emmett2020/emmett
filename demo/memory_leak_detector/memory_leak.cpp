#include <iostream>

namespace {
  void malloc_without_free() {
    // NOLINTNEXTLINE
    auto *m = static_cast<int *>(::malloc(2'048));
    std::cout << *m << "\n";
  }
} // namespace

auto main() -> int {
  malloc_without_free();
  return 0;
}
