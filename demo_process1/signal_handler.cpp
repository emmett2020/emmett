//
// Created by 张乃港 on 2021/5/12.
//

#include "signal_handler.h"
//
//void SignalHandler::SigHandler(int sig) {
//    if (sig == SIGCHLD) {
//        waitpid(m_pid, &m_ret_stat, WNOHANG);
//    } else if (sig == SIGALRM) { ; // We will ignore SIGALRM
//    } else if (sig == SIGINT) {
//        std::cout << "Interrupt" << std::endl;
//    }
//}
//
//bool SignalHandler::is_chld_ret_success() {
//    return WIFEXITED(m_ret_stat);
//}