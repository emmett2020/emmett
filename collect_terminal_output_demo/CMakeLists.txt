cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
PROJECT(collect_terminal_output)
include_directories()

add_executable(collect_terminal_output redirect.cpp redirect.cpp)
