#pragma once

#define FMT_HEADER_ONLY
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>
#include <vector>
#include "vars.h"
using json = nlohmann::json;

void ExecveCommand(const std::string& commandStr) {
  auto logger = spdlog::get("basic_logger");
  std::stringstream ss(commandStr);
  std::string path;
  std::string command;
  std::string temp;

  int fd = open("CommandOutputFile.txt", O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
  if (fd == -1) {
    logger->error("Can't open file.");
    return;
  }

  dup2(fd, 1);
  dup2(fd, 2);
  close(fd);

  char* envp[] = {(char*)"PATH=/bin", nullptr};
  char* argv[10];
  int idx = 0;
  while (ss >> temp) {
    argv[idx++] = (char*)temp.c_str();
  }
  argv[idx] = nullptr;
  execve("/bin/ls", argv, envp);
}

std::string GetExecveStrRes() {
  auto logger = spdlog::get("basic_logger");
  std::ifstream ifs;
  ifs.open("CommandOutputFile.txt", std::ios::in);

  if (!ifs.is_open()) {
    logger->error("Can't open such file!");
    return "";
  }
  std::string buf;
  std::string temp;
  while (std::getline(ifs, temp)) {
    buf += temp + "\r\n";
  }
  return buf;
}

std::vector<json> StrResToJson(const std::string& res) {
  auto logger = spdlog::get("basic_logger");
  std::vector<LSResStruct> LSSVec;
  LSResStruct LSS;
  std::stringstream ss(res);
  std::string line, field;
  std::vector<json> resJSON;
  ss >> line;  // ignore first line
  ss >> line;
  while (getline(ss, line)) {
    logger->info(line);

    std::vector<std::string> temp;
    std::stringstream sss(line);

    while (ss >> field) {
      temp.push_back(field);
    }
    LSS.perm = temp[0];
    LSS.cnt = temp[1];
    LSS.size = temp[2];
    LSS.uid = temp[3];
    LSS.gid = temp[4];
    LSS.size = temp[5];
    LSS.data = temp[6];
    LSS.name = temp[7];
    LSSVec.push_back(LSS);
  }
  int idx = 0;
  for (auto v : LSSVec) {
    resJSON[idx++] = v;
  }
  return resJSON;
}
