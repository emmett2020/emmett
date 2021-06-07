/*
 * @author: xiaomingZhang2020@outlook.com
 * @copyright: GPL
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <mutex>
#include <shared_mutex>

// 多线程读写同一文件网上C++代码并不多
// 参考自己的链接：https://rwz6w0spa4.feishu.cn/wiki/wikcnlYVX1LlXlOmZyMJcY6rrcc
// 本版本是多用户读同一个文件，单用户可写，同时读的需求明显大于写的需求
// 采用的是C++17的共享锁，请确保c编译器版本大于17

// 对于多线程读大文件的case
// 通过seekg定位阅读位置，每个线程分别定位到【0，1/3 * len(file)，2/3 * len(file)】
// 使用三个线程，同时读同一文件的不同部分，最后整合

// 对于多用户（一个用户一个线程）读同一文件的case（每个用户读同一文件的全部）


class ThreadSafeRWSameFile {// thread_rw_lock
public:
    explicit ThreadSafeRWSameFile(std::string &file_path) : m_file_path(file_path), m_pos(0) {}

    void readline() {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        std::fstream file(m_file_path.c_str(), std::ios::in);
        unsigned int pos = 0;
        while (file.good()) {
            std::string line;
            file.seekg(pos, std::ios::beg);
            getline(file, line);
            std::cout << std::this_thread::get_id() <<line << std::endl;
            pos += line.size() + 1; // 加1是为了略过文件末尾的'\n'，否则会在读取第一行后死循环了
        }
    }

    void writeline(const std::string& line){
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        std::fstream file(m_file_path.c_str(), std::ios::app);
        file<<std::endl<<line;
    }

private:
    mutable std::shared_mutex m_mutex;
    std::string m_file_path;
    unsigned int m_pos;
};

int main()
{
    std::string filepath = "../sample.txt";
    ThreadSafeRWSameFile sinker(filepath);

    std::vector<std::thread> reader_vec;
    for(int i = 0; i < 2; ++i){
        reader_vec.push_back(std::thread(&ThreadSafeRWSameFile::readline,std::ref(sinker)));
    }
    std::thread writer = std::thread(&ThreadSafeRWSameFile::writeline,std::ref(sinker),"hello");
    for(auto & i : reader_vec)
        i.join();
    writer.join();
    getchar();
    return 0;
}