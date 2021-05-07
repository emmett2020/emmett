cmake_minimum_required(VERSION 3.26)
project(design_pattern)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)

add_executable(01_simple_factory 01_simple_factory.cpp)
add_executable(02_factory_method 02_factory_method.cpp)
