cmake_minimum_required(VERSION 3.17)

project(read_file)
set(CMAKE_CXX_STANDARD  20)
set(CMAKE_BUILD_TYPE Debug)

include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_libraries(/opt/homebrew/Cellar/boost/1.75.0_2/lib)

add_executable(read_file main.cpp main.cpp)