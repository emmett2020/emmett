namespace {
  void allocate_free() {
    // NOLINTNEXTLINE
    auto *m = new int[2'048];
    // NOLINTNEXTLINE
    delete (m);
  }
} // namespace

auto main() -> int {
  allocate_free();
  return 0;
}
