cmake_minimum_required(VERSION 3.17)

project(CommandOperator)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

# boost
include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_directories(/opt/homebrew/Cellar/boost/1.75.0_2/lib)

# spdlog
include_directories(/opt/homebrew/Cellar/spdlog/1.8.2/include)
link_libraries(/opt/homebrew/Cellar/spdlog/1.8.2/lib)


# nlohmann-json
include_directories(/opt/homebrew/Cellar/nlohmann-json/3.9.1_1/include)
link_libraries(/opt/homebrew/Cellar/nlohmann-json/3.9.1_1/lib)

# fmt
include_directories(/opt/homebrew/Cellar/fmt/7.1.3/include)
link_libraries(/opt/homebrew/Cellar/fmt/7.1.3/lib)

set(SRC_LIST vars.h exec_cmd.h ReportOperator.h  recv_cmd.h print_cmd_res.h)

add_executable(CommandOperator main.cpp ${SRC_LIST})
#add_executable(myProcess test.cpp ${SRC_LIST})