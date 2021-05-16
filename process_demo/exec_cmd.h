#ifndef MYPROCESS_EXEC_CMD_H
#define MYPROCESS_EXEC_CMD_H
#define FMT_HEADER_ONLY
#include <boost/interprocess/ipc/message_queue.hpp>
#include <iostream>
#include <unistd.h>
#include <thread>
#include <vector>
#include <functional>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include "vars.h"
#include <algorithm>
using json = nlohmann::json;


void ExecveCommand(const std::string &commandStr) {
    /*
     * 目前仅仅支持ls命令，后续改进成支持所有命令
     */
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

    // 输出重定向,后期改为unix域通信
    dup2(fd, 1);
    dup2(fd, 2);
    close(fd);

//    ss >> path;
    char *envp[] = {(char *) "PATH=/bin", nullptr};
    char *argv[10];
    int idx = 0;
    while (ss >> temp) {
        argv[idx++] = (char *) temp.c_str();
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


std::vector<json> StrResToJson(const std::string& res){
    /*
     * 这里处理的也不好，json就json，不应该用vector来装
     */
    auto logger = spdlog::get("basic_logger");
    std::vector<LSResStruct> LSSVec;
    LSResStruct LSS;
    std::stringstream ss(res);
    std::string line, field;
    std::vector<json> resJSON;
    ss >> line; // ignore first line
    ss >> line;
    while(getline(ss,line))
    {
        logger->info(line);

        std::vector<std::string> temp;
        std::stringstream sss(line);

        while(ss >> field) {
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
    for(auto v : LSSVec)
    {
        resJSON[idx++] = v;
    }
    return resJSON;
}

#endif //MYPROCESS_EXEC_CMD_H


