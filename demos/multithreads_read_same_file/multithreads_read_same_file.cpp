#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// Assume sample.txt is a very large file, try to read it uses multiply threads.
// Firstly,  Use `seekg` to locate the position to read, thus each thread is
// located to [0, 1/3 * len(file), 2/3 * len(file]. Then print the result.

// WARNING: DOING.

// thread_rw_lock
class ThreadSafeRWSameFile {
public:
  explicit ThreadSafeRWSameFile(std::string file_path)
      : file_path_(std::move(file_path)) {}

  void readline() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!std::filesystem::is_regular_file(file_path_)) {
      throw std::runtime_error("file path error");
    }

    std::fstream file(file_path_, std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("Open file error");
    }

    unsigned int pos = 0;
    std::string line;
    while (file.good()) {
      file.seekg(pos, std::ios::beg);
      std::getline(file, line);
      std::cout << std::this_thread::get_id() << ": " << line << std::endl;
      // skip the '\n' at the end of the file, otherwise it will loop forever
      pos += line.size() + 1;
    }
  }

  void writeline(const std::string &line) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::fstream file(file_path_, std::ios::app);
    file << "\n" << line;
  }

private:
  mutable std::shared_mutex mutex_;
  std::string file_path_;
  unsigned pos_{0};
};

int main() {
  std::string file_path = "../sample.txt";
  ThreadSafeRWSameFile sinker(file_path);

  std::vector<std::thread> readers;
  readers.reserve(2);
  for (int i = 0; i < 2; ++i) {
    readers.emplace_back(&ThreadSafeRWSameFile::readline, std::ref(sinker));
  }
  auto writer =
      std::thread(&ThreadSafeRWSameFile::writeline, std::ref(sinker), "hello");
  for (auto &reader : readers) {
    reader.join();
  }
  writer.join();
  return 0;
}
