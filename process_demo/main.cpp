#define FMT_HEADER_ONLY
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include "vars.h"
#include "recv_cmd.h"
#include "exec_cmd.h"
#include "print_cmd_res.h"
#include <nlohmann/json.hpp>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>

using namespace std;
using json = nlohmann::json;

[[noreturn]] void MainOperator() {
    string commandStr;
    string execResStr;
    json execResJSON;
    pid_t pid;
    int ret_status;
    while (true) {
        if (!(commandStr = TryRecvCommandFromChildProcess()).empty()) {
            // 执行命令
            if ((pid = fork()) == 0) {
                ExecveCommand(commandStr);
            } else {
                wait(&ret_status);
                // 获取命令执行的json结果，后续优化成用信号
                std::this_thread::sleep_for(std::chrono::duration<double>(0.5));
                execResStr = GetExecveStrRes();
                execResJSON = StrResToJson(execResStr);
                // 将命令执行的json结果，美化控制台打印，
                PrintJsonResToConsole(execResJSON);
                // 将命令执行的json结果，存入数据库
            }
        }
    }

}

auto getLogger() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back(std::make_shared<spdlog::sinks::daily_file_sink_mt>("../logs/commandOperator.log", 12, 00));
    auto logger = make_shared<spdlog::logger>("basic_logger", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);
    sinks[1]->set_pattern("[%Y-%m-%d %T][%-7l][%-20@]%v");
    sinks[0]->set_pattern("[%T][%-7l]%v");
    logger->flush_on(spdlog::level::err);
    spdlog::register_logger(logger);
    spdlog::flush_every(std::chrono::seconds(3));
    return logger;
}


int main() {
    pid_t exec_pid, report_pid;
    auto logger = getLogger();
    if ((exec_pid = fork()) == 0) // 创建执行子进程
    {
        RecvCommandFromUser();
    } else {
        if ((report_pid = fork()) == 0) // 创建上报数据库子进程
        {

        } else { // 这里是主进程的逻辑
            MainOperator();
        }
    }


    return 0;
}

