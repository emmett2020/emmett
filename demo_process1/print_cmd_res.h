#ifndef MYPROCESS_PRINT_CMD_RES_H
#define MYPROCESS_PRINT_CMD_RES_H
#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <nlohmann/json.hpp>
#include <sstream>
using json = nlohmann::json;

void PrintJsonResToConsole(const json& resJSON){
    auto console_logger = spdlog::stdout_color_mt("console_logger");
    console_logger->info(resJSON);
}

#endif //MYPROCESS_PRINT_CMD_RES_H
