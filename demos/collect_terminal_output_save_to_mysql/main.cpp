#define FMT_HEADER_ONLY
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include "exec_cmd.h"
#include "print_cmd_res.h"
#include "recv_cmd.h"
#include "vars.h"

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
      if ((pid = fork()) == 0) {
        ExecveCommand(commandStr);
      } else {
        wait(&ret_status);
        std::this_thread::sleep_for(std::chrono::duration<double>(0.5));
        execResStr = GetExecveStrRes();
        execResJSON = StrResToJson(execResStr);
        PrintJsonResToConsole(execResJSON);
      }
    }
  }
}

auto getLogger() {
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  sinks.push_back(std::make_shared<spdlog::sinks::daily_file_sink_mt>(
      "../logs/commandOperator.log", 12, 00));
  auto logger =
      make_shared<spdlog::logger>("basic_logger", sinks.begin(), sinks.end());
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
  if ((exec_pid = fork()) == 0) {
    RecvCommandFromUser();
  } else {
    if ((report_pid = fork()) == 0) {
    } else {
      MainOperator();
    }
  }

  return 0;
}
