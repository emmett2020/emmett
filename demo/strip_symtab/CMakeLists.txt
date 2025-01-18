cmake_minimum_required(VERSION 3.26)
project(emmett)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")

set(emmett_path ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include(${emmett_path}/cmake/ReserveDebugSymbolAndStripSymtab.cmake)

add_library(foo SHARED foo.cpp)
reserve_debug_symbol_then_strip_symtab(foo
                                       libfoo.so
                                       ${CMAKE_CURRENT_SOURCE_DIR}/build
                                       ${CMAKE_CURRENT_SOURCE_DIR}/build/sym)

add_executable(main main.cpp)
target_link_libraries(main foo)

