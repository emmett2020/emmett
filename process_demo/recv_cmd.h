//
// Created by 张乃港 on 2021/5/12.
//

#ifndef MYPROCESS_RECV_CMD_H
#define MYPROCESS_RECV_CMD_H
#define FMT_HEADER_ONLY

#include <boost/interprocess/ipc/message_queue.hpp>
#include <iostream>
#include <unistd.h>
#include <thread>
#include <vector>
#include <functional>
#include <spdlog/spdlog.h>

int RecvCommandFromUser() {
    auto logger = spdlog::get("basic_logger");
    std::string data;
    try {
        //open a message queue, parent should create mq ok before child process open it
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, "command_queue",MQ_MSG_NUM,MQ_MSG_MAX_SIZE);
        //send message
        for (;;) {
            std::cout << "Please input the COMMAND you want to execute: ";
            std::getline(std::cin, data);
            logger->debug(data);
            mq.send(data.c_str(), data.size() + 1, 0);
            std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
        }
    }
    catch (boost::interprocess::interprocess_exception &e) {
        logger->error(e.what());
        return -1;
    }
}

std::string TryRecvCommandFromChildProcess() {
    auto logger = spdlog::get("basic_logger");
    try {
        //open a message queue
        char CommandCStr[MQ_MSG_MAX_SIZE];
        std::string commandStr;
        unsigned priority;
        boost::interprocess::message_queue::size_type received_size;
        boost::interprocess::message_queue mq(boost::interprocess::open_or_create, "command_queue", MQ_MSG_NUM,
                                              MQ_MSG_MAX_SIZE);
        if (mq.try_receive(CommandCStr, sizeof(CommandCStr), received_size, priority)) {
            logger->debug("received size: {}", received_size);
            commandStr = std::string(CommandCStr);
            memset(CommandCStr, 0, sizeof(CommandCStr));
        }
        logger->debug(commandStr);
        return commandStr;
    }
    catch (boost::interprocess::interprocess_exception &e) {
        logger->error(e.what());
        throw;
    }

}

#endif //MYPROCESS_RECV_CMD_H
