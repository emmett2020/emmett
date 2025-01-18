cmake_minimum_required(VERSION 3.13)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)

set(emmett_path ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include(${emmett_path}/cmake/FetchBenchmark.cmake)

add_executable(memory_allocation memory_allocation.cpp)
target_link_libraries(memory_allocation benchmark::benchmark)
