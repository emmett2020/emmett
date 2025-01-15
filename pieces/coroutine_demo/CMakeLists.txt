cmake_minimum_required(VERSION 3.26)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-fcoroutines ${CMAKE_CXX_FLAGS}")

add_executable(main main.cpp)
