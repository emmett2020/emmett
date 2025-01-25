cmake_minimum_required(VERSION 3.26)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

add_executable(memory_leak memory_leak.cpp)
add_executable(safely_allocate_memory safely_allocate_memory.cpp)

if (ENABLE_ASAN)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g -O1 ${CMAKE_CXX_FLAGS}")
endif()
