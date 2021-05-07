#include <coroutine>
#include <functional>
#include <mutex>
#include <print>
#include <queue>
#include <thread>

using namespace std; // NOLINT

int cnt = 1;
mutex m;                  // NOLINT
queue<int> request_queue; // NOLINT

struct Task {
  struct promise_type {
    auto initial_suspend() { return suspend_never{}; }

    auto final_suspend() noexcept { return suspend_never{}; }

    void unhandled_exception() { terminate(); }

    auto get_return_object() { return Task{}; }

    auto return_void() {}
  };
};

void AddOneByCallback(const function<void(void)> &f) {
  request_queue.push(1);
  f();
}

struct AddOneByAwaiter {
  auto await_ready() { return false; }

  auto await_resume() {}

  auto await_suspend(coroutine_handle<> h) {
    auto f = [h]() mutable { h.resume(); };
    AddOneByCallback(f);
  }
};

Task execute(int n) {
  this_thread::sleep_for(chrono::seconds(1));
  m.lock();
  for (int i = 0; i < n; ++i) {
    request_queue.push(1);
  }
  m.unlock();
  while ((n--) != 0) {
    co_await AddOneByAwaiter();
  }
}

int main() {
  execute(1000);
  return 0;
}
