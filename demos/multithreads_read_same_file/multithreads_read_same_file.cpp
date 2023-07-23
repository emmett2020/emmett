#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <vector>

// For multi-threaded read large file case.
// Use `seekg` to locate the reading position, and each thread is located to
//          [0, 1/3 * len(file), 2/3 * len(file]
// Uses three threads, reading different parts of the same file at the same
// time, and finally integrates

class ThreadSafeRWSameFile {  // thread_rw_lock
 public:
  explicit ThreadSafeRWSameFile(std::string& file_path)
      : m_file_path(file_path), m_pos(0) {}

  void readline() {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    std::fstream file(m_file_path.c_str(), std::ios::in);
    unsigned int pos = 0;
    while (file.good()) {
      std::string line;
      file.seekg(pos, std::ios::beg);
      getline(file, line);
      std::cout << std::this_thread::get_id() << line << std::endl;
      // skip the '\n' at the end of the file, otherwise it will
      // loop dead after reading the first line
      pos += line.size() + 1;
    }
  }

  void writeline(const std::string& line) {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    std::fstream file(m_file_path.c_str(), std::ios::app);
    file << std::endl << line;
  }

 private:
  mutable std::shared_mutex m_mutex;
  std::string m_file_path;
  unsigned int m_pos;
};

int main() {
  std::string filepath = "../sample.txt";
  ThreadSafeRWSameFile sinker(filepath);

  std::vector<std::thread> reader_vec;
  for (int i = 0; i < 2; ++i) {
    reader_vec.push_back(
        std::thread(&ThreadSafeRWSameFile::readline, std::ref(sinker)));
  }
  std::thread writer =
      std::thread(&ThreadSafeRWSameFile::writeline, std::ref(sinker), "hello");
  for (auto& i : reader_vec)
    i.join();
  writer.join();
  getchar();
  return 0;
}