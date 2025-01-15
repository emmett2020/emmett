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

#include <array>
#include <iostream>
#include <thread>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>

using namespace std::chrono_literals;

namespace {
  auto message_queue_a() -> int {
    try {
      // Erase previous message queue
      auto message = std::array<char, 100>{};

      // mq exits even though process fail or end, so remove it first.
      boost::interprocess::message_queue::remove("message_queue");

      // Creat a message queue.
      auto mq = boost::interprocess::message_queue{
        boost::interprocess::create_only,
        "message_queue",
        100,
        sizeof(message)};

      // send message
      auto data = std::string{};
      for (;;) {
        std::cout << "Please input the COMMAND you want to execute: ";
        std::getline(std::cin, data);
        std::memcpy(message.data(), data.c_str(), data.size());
        mq.send(static_cast<const void *>(message.data()), sizeof(message), 0);
        std::memset(message.data(), 0, sizeof(message));
        std::this_thread::sleep_for(0.01s);
      }
      return 0;
    } catch (boost::interprocess::interprocess_exception &e) {
      std::cout << e.what() << '\n';
      return 1;
    }
  }

  auto message_queue_b() -> int {
    try {
      // Open a message queue.
      auto message = std::array<char, 100>{};
      auto mq = boost::interprocess::message_queue{boost::interprocess::open_only, "message_queue"};
      auto priority      = 0U;
      auto received_size = boost::interprocess::message_queue::size_type{0};

      // receive
      while (true) {
        mq.receive(message.data(), sizeof(message), received_size, priority);
        std::cout << "Received Message: {}" << message.data();
        std::memset(message.data(), 0, sizeof(message));
      }
      return 0;
    } catch (boost::interprocess::interprocess_exception &e) {
      std::cout << e.what() << '\n';
      return 1;
    }
  }
} // namespace

auto main() -> int {
  auto pid = ::fork();
  if (pid == 0) {
    std::this_thread::sleep_for(1s);
    message_queue_b();
  } else {
    message_queue_a();
  }
}
