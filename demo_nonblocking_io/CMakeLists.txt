cmake_minimum_required(VERSION 3.17)



project(nonblockingIO)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_directories(/opt/homebrew/Cellar/boost/1.75.0_2/lib)

add_executable(nonblockingIO use_this_one.cpp)