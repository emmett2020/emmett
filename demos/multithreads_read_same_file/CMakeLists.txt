cmake_minimum_required(VERSION 3.26)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)

add_executable(main multithreads_read_same_file.cpp)

# https://cmake.org/cmake/help/v3.29/module/FindBoost.html
# find_package(Boost REQUIRED)
# target_link_libraries(main ${Boost_LIBRARIES})
