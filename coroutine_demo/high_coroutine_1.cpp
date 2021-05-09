#include <iostream>
#include <experimental/coroutine>
#include <thread>
#include <chrono>
#include <functional>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <queue>

using namespace std;
using namespace std::experimental;
using call_back = std::function<void(int)>;

/*
 * 本代码是多线程+生产者消费者版
 * 使用条件变量通知有任务可用
 * 使用信号量记录可用资源个数
 * 使用锁对cnt进行保护
 * 线程使用C++11标准库
 */


long long cnt = 1; // 增加他

queue<int> request_queue; //任务区

mutex m; // 锁

class ThreadPool {
public:
    explicit ThreadPool(int n) : thread_num(n),done(false) {
        for (int i = 0; i < thread_num; ++i) {
            threadpool.emplace_back(&ThreadPool::run,this);
        }

    }

    void run() { // 实际处理函数
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
        // 内存回收
        done = true;
        for(int i = 0; i < thread_num; ++i)
            threadpool[i].join();
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
        auto elapsedTime = std::chrono::duration<double>(cur_time - old_time).count();
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
    auto elapsedTime = std::chrono::duration<double>(end_time - start_time).count();
    cout << "total cost: " << elapsedTime;

    return 0;
}
