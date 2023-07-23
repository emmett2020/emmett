#pragma once
#define FMT_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sstream>
using json = nlohmann::json;

void PrintJsonResToConsole(const json& resJSON) {
  auto console_logger = spdlog::stdout_color_mt("console_logger");
  console_logger->info(resJSON);
}
