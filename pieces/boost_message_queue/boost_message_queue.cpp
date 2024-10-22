#include <array>
#include <iostream>
#include <print>
#include <thread>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>

namespace interprocess = boost::interprocess;
using message_queue = boost::interprocess::message_queue;
using namespace std::chrono_literals;

int MessageQueueA() {
  try {
    // Erase previous message queue
    std::array<char, 100> message{};

    // mq exits even though process fail or end, so remove it first.
    interprocess::message_queue::remove("message_queue");

    // Creat a message queue.
    interprocess::message_queue mq{interprocess::create_only, "message_queue",
                                   100, sizeof(message)};

    // send message
    std::string data;
    for (;;) {
      std::println("Please input the COMMAND you want to execute: ");
      std::getline(std::cin, data);
      std::memcpy(message.data(), data.c_str(), data.size());
      mq.send(static_cast<const void *>(message.data()), sizeof(message), 0);
      std::memset(message.data(), 0, sizeof(message));
      std::this_thread::sleep_for(0.01s);
    }
    return 0;
  } catch (interprocess::interprocess_exception &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
}

int MessageQueueB() {
  try {
    // Open a message queue.
    std::array<char, 100> message{};
    message_queue mq{interprocess::open_only, "message_queue"};
    unsigned priority = 0;
    message_queue::size_type received_size = 0;

    // receive
    while (true) {
      mq.receive(message.data(), sizeof(message), received_size, priority);
      std::string data(message.data());
      std::println("Received Message: {}", message.data());
      std::memset(message.data(), 0, sizeof(message));
    }
    return 0;
  } catch (interprocess::interprocess_exception &e) {
    std::print("{}", e.what());
    return 1;
  }
}

int main() {
  pid_t pid = ::fork();
  if (pid == 0) {
    std::this_thread::sleep_for(1s);
    MessageQueueB();
  } else {
    MessageQueueA();
  }
}
