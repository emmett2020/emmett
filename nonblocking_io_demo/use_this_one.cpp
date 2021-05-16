#include <boost/interprocess/ipc/message_queue.hpp>
#include <iostream>
#include <unistd.h>
#include <thread>
using namespace std;
using namespace boost::interprocess;



int MessageQueue_A() {
    try {
        //erase previous message queue
        char message[100];
        message_queue::remove("message_queue"); // mq exits even though process fail or end
        //creat a message queue
        message_queue mq(create_only, "message_queue", 100, sizeof(message));
        //send message
        for (;;) {
            std::cout << "Please input the COMMAND you want to execute: ";
            std::string data;
            std::getline(std::cin, data);
            memcpy(message,data.c_str(),data.size());
            mq.send((const void *) message, sizeof(message), 0);
            memset(message,0,sizeof(message));
            this_thread::sleep_for(chrono::duration<double>(0.01));
        }
    }
    catch (interprocess_exception &e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int MessageQueue_B() {
    try {
        //open a message queue
        char messageStr[100];
        message_queue mq(open_only, "message_queue");
        unsigned priority;
        message_queue::size_type received_size;

        //receive
        while (true) {
            mq.receive(messageStr,sizeof(messageStr) , received_size, priority);
            string data(messageStr);
            std::cout << "Receive Message:" << data << std::endl;
            memset(messageStr,0,sizeof(messageStr));
        }

    }
    catch (interprocess_exception &e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
}


int main() {
    pid_t pid;
    if ((pid = fork()) == 0) {
        sleep(1);
        MessageQueue_B();
    } else {
        MessageQueue_A();
    }

}