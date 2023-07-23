#include <chrono>
#include <experimental/coroutine>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

using namespace std;
using namespace std::experimental;
using call_back = std::function<void(int)>;

// This code is multithreaded + producer consumer edition
// Use condition variables to notify that a task is available
// Record the number of available resources using semaphore
// Use locks to protect cnt
// Threads use the C++11 standard library

long long cnt = 1;
queue<int> request_queue;
mutex m;

class ThreadPool {
 public:
  explicit ThreadPool(int n) : thread_num(n), done(false) {
    for (int i = 0; i < thread_num; ++i) {
      threadpool.emplace_back(&ThreadPool::run, this);
    }
  }

  void run() {
    while (!done) {
      m.lock();
      if (!request_queue.empty()) {
        int n = request_queue.front();
        request_queue.pop();
        this_thread::sleep_for(chrono::seconds(1));
        cnt += n;
      }
      m.unlock();
    }
  }

  ~ThreadPool() {
    done = true;
    for (int i = 0; i < thread_num; ++i) {
      threadpool[i].join();
    }
  }

 private:
  int thread_num;
  atomic_bool done;
  vector<thread> threadpool;
};

int main() {
  cout << "hello world" << endl;
  auto start_time = std::chrono::steady_clock::now();
  ThreadPool tp{6};

  auto old_time = std::chrono::steady_clock::now();
  while (true) {
    auto cur_time = std::chrono::steady_clock::now();
    auto elapsedTime =
        std::chrono::duration<double>(cur_time - old_time).count();
    if (elapsedTime >= 1) {
      old_time = cur_time;
      m.lock();
      if (cnt >= 1000) {
        m.unlock();
        break;
      }
      request_queue.push(1);
      request_queue.push(1);
      request_queue.push(1);
      request_queue.push(1);
      request_queue.push(1);
      request_queue.push(1);
      m.unlock();
    }
  }
  auto end_time = std::chrono::steady_clock::now();
  auto elapsedTime =
      std::chrono::duration<double>(end_time - start_time).count();
  cout << "total cost: " << elapsedTime;

  return 0;
}
