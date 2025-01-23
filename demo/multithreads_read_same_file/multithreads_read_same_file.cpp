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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// TODO: Still doing and needs test real speed.

// Assume sample.txt is a very large file, try to read it uses multiply threads.
// Firstly,  Use `seekg` to locate the position to read, thus each thread is
// located to [0, 1/3 * len(file), 2/3 * len(file]. Then print the result.

// thread_rw_lock
class thread_safe_read_write_file {
public:
  explicit thread_safe_read_write_file(std::string file_path)
    : file_path_(std::move(file_path)) {
  }

  void readline() {
    auto lock = std::shared_lock<std::shared_mutex>(mutex_);

    if (!std::filesystem::is_regular_file(file_path_)) {
      throw std::runtime_error("file path error");
    }

    auto file = std::fstream{file_path_, std::ios::in};
    if (!file.is_open()) {
      throw std::runtime_error("Open file error");
    }

    auto pos  = uint32_t{0};
    auto line = std::string{};
    while (file.good()) {
      file.seekg(pos, std::ios::beg);
      std::getline(file, line);
      std::cout << std::this_thread::get_id() << ": " << line << '\n';
      // skip the '\n' at the end of the file, otherwise it will loop forever
      pos += line.size() + 1;
    }
  }

  void writeline(const std::string &line) {
    auto lock = std::unique_lock<std::shared_mutex>{mutex_};
    auto file = std::fstream{file_path_, std::ios::app};
    file << "\n" << line;
  }

private:
  mutable std::shared_mutex mutex_;
  std::string file_path_;
  uint32_t pos_{0};
};

int main() {
  auto file_path = std::string{"./sample.txt"};
  auto sinker    = thread_safe_read_write_file{file_path};

  auto readers = std::vector<std::thread>{};
  readers.reserve(2);
  for (int i = 0; i < 2; ++i) {
    readers.emplace_back(&thread_safe_read_write_file::readline, std::ref(sinker));
  }
  auto writer = std::thread(&thread_safe_read_write_file::writeline, std::ref(sinker), "hello");
  for (auto &reader: readers) {
    reader.join();
  }
  writer.join();
  return 0;
}

