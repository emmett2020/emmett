#pragma once

#include <functional>

// TODO: Use std::scope_exit.
class scope_guard {
public:
  explicit scope_guard(std::function<void()> on_exit)
      : on_exit_(std::move(on_exit)) {}

  ~scope_guard() {
    if (on_exit_) {
      on_exit_();
    }
  }

  scope_guard(const scope_guard &) = delete;
  scope_guard &operator=(const scope_guard &) = delete;

  scope_guard(scope_guard &&other) noexcept
      : on_exit_(std::move(other.on_exit_)) {
    other.on_exit_ = nullptr;
  }

  scope_guard &operator=(scope_guard &&other) noexcept {
    if (this != &other) {
      on_exit_ = std::move(other.on_exit_);
      other.on_exit_ = nullptr;
    }
    return *this;
  }

private:
  std::function<void()> on_exit_;
};
