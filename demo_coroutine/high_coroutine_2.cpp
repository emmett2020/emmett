/*
 * 现在我们使用协程 + 线程池来实现
 */

#include <vector>
#include <iostream>
#include <thread>
#include <experimental/coroutine>
#include <queue>
#include <mutex>

using namespace std;
using namespace experimental;

auto start_time = chrono::steady_clock::now();

int cnt = 1;
queue<int> request_queue;
mutex m;

class ThreadPool{
public:
    explicit ThreadPool(int n):thread_num(n),done(false){
        for(int i = 0; i < thread_num; ++i)
        {
            threadpool.emplace_back(&ThreadPool::run,this);
        }
    }

    void run(){
        while(!done)
        {
            m.lock();
            if(!request_queue.empty()){
                int tmp = request_queue.front();
                request_queue.pop();
                this_thread::sleep_for(chrono::seconds(1));
                cnt += tmp;
                if (cnt == 1001){
                    auto end_time = chrono::steady_clock::now();
                    auto elapsed_time = chrono::duration<double>(end_time - start_time).count();
                    cout<<"total cost: "<<elapsed_time<<endl;
                }
            }
            m.unlock();
        }
    }


    ~ThreadPool(){
        done = true;
        for(int i = 0; i < thread_num; ++i)
            threadpool[i].join();
    }

private:
    int thread_num;
    vector<thread> threadpool;
    atomic_bool done;
};



ThreadPool pool(6);

struct Task {
    struct promise_type {
        auto initial_suspend() { return suspend_never{}; }
        auto final_suspend() noexcept { return suspend_never{}; }
        void unhandled_exception(){terminate();}
        auto get_return_object(){return Task{};}
        auto return_void(){}
    };
};


void AddOneByCallback(const function<void(void)>& f)
{
    request_queue.push(1);
    f();
}

struct AddOneByAwaiter{
    auto await_ready(){return false;}
    auto await_resume(){}
    auto await_suspend(coroutine_handle<> h){
        auto f = [h]()mutable{
            h.resume();
        };
        AddOneByCallback(f);
    }
};

Task execute(int n){

    this_thread::sleep_for(chrono::seconds(1));
    m.lock();
    for(int i = 0; i < n; ++i) {
        request_queue.push(1);
    }
    m.unlock();
    while(n--) {
        co_await AddOneByAwaiter();
    }
}

int main()
{
    cout<<"hello world"<<endl;
    execute(1000);
    getchar();
    return 0;
}





