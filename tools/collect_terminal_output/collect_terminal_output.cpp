#include <fcntl.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

void ExecveCommand(const std::string& command_str) {
  // NOLINTBEGIN
  std::stringstream sss{command_str};
  std::string path;
  std::string command;
  std::string temp;

  int descriptor =
      ::open("CommandOutputFile.txt", O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC,
             S_IRWXO | S_IRWXU | S_IRWXG | S_IWUSR | S_IWOTH | S_IWGRP);
  if (descriptor == -1) {
    std::cout << "open CommandOutputFile Error" << std::endl;
    return;
  }

  std::cout << descriptor << std::endl;
  dup2(descriptor, 1);
  dup2(descriptor, 2);
  close(descriptor);

  sss >> path;
  char* envp[] = {(char*)"PATH=/bin", nullptr};
  char* argv[10];
  int idx = 0;
  while (sss >> temp) {
    argv[idx++] = (char*)temp.c_str();
  }
  argv[idx] = nullptr;
  execve(path.c_str(), argv, envp);
  // NOLINTEND
}

auto main() -> int {
  std::string command("/bin/ls ls -al");
  if ((fork()) == 0) {
    ExecveCommand(command);
  } else {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Start to output: " << std::endl;
    std::ifstream ifs;
    ifs.open("CommandOutputFile.txt", std::ios::in);

    if (!ifs.is_open()) {
      std::cout << "read fail." << std::endl;
      return -1;
    }
    std::string buf;
    while (std::getline(ifs, buf)) {
      std::cout << buf << std::endl;
    }
  }
  return 0;
}