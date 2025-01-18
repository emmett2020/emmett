cmake_minimum_required(VERSION 3.26)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)

find_package(Boost REQUIRED CONFIG COMPONENTS filesystem system regex program_options)

add_executable(main multithreads_read_same_file.cpp)
