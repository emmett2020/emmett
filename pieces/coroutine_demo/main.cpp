/*
 * Copyright (c) 2024 Emmett Zhang
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <coroutine>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace {
  int cnt            = 1;                 // NOLINT
  auto m             = std::mutex{};      // NOLINT
  auto request_queue = std::queue<int>{}; // NOLINT

  struct Task {
    struct promise_type {
      auto initial_suspend() {
        return std::suspend_never{};
      }

      auto final_suspend() noexcept {
        return std::suspend_never{};
      }

      void unhandled_exception() {
        std::terminate();
      }

      auto get_return_object() {
        return Task{};
      }

      auto return_void() {
      }
    };
  };

  void add_one_by_callback(const std::function<void(void)> &f) {
    request_queue.push(1);
    f();
  }

  struct add_one_by_awaiter {
    auto await_ready() {
      return false;
    }

    auto await_resume() {
    }

    auto await_suspend(std::coroutine_handle<> h) {
      auto f = [h]() mutable {
        h.resume();
      };
      add_one_by_callback(f);
    }
  };

  auto execute(int n) -> Task {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    m.lock();
    for (int i = 0; i < n; ++i) {
      request_queue.push(1);
    }
    m.unlock();
    while ((n--) != 0) {
      co_await add_one_by_awaiter();
    }
  }

} // namespace

auto main() -> int {
  execute(1'000);
  return 0;
}
