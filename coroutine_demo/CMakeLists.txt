cmake_minimum_required(VERSION 3.17)
project(coroutine)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

add_executable(coroutine high_coroutine_2.cpp)