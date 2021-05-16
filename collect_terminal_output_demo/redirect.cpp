#include <iostream>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

void ExecveCommand(const std::string &commandStr) {
    std::stringstream ss(commandStr);
    std::string path;
    std::string command;
    std::string temp;

    int fd = open("CommandOutputFile.txt", O_CREAT | O_RDWR | O_TRUNC, S_IRWXO | S_IRWXU | S_IRWXG | S_IWUSR | S_IWOTH | S_IWGRP);
    if(fd == -1) {
        std::cout<<fd<<std::endl;
        return;
    }


    std::cout<<fd<<std::endl;
    dup2(fd,1);
    dup2(fd,2);
    close(fd);

    ss >> path;
    char *envp[] = {(char *) "PATH=/bin", nullptr};
    char *argv[10];
    int idx = 0;
    while (ss >> temp) {
        argv[idx++] = (char *) temp.c_str();
    }
    argv[idx] = nullptr;
    execve(path.c_str(), argv, envp);
}

int main() {
    std::string s("/bin/ls ls -al");
    if ((fork()) == 0) {
        ExecveCommand(s);
    }else {
        sleep(1);
        std::cout << "输出开始：" << std::endl;
        std::ifstream ifs;
        ifs.open("CommandOutputFile.txt", std::ios::in);

        if (!ifs.is_open()) {
            std::cout << "read fail." << std::endl;
            return -1;
        }
        std::cout << "test 1" << std::endl;
        std::string buf;
        while (std::getline(ifs, buf)) {
            std::cout << "test" << std::endl;
            std::cout << buf << std::endl;
        }
    }
    return 0;
}