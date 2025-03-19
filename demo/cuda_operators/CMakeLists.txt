cmake_minimum_required(VERSION 3.26)
project(emmett LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

include(cmake/FetchCatch2.cmake)
include(cmake/FetchLibtorch.cmake)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

file(GLOB_RECURSE test_precision_files "test_precision/*.cu")
add_executable(test_precision ${test_precision_files})

set_target_properties(test_precision PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_precision PRIVATE
                                     ${TORCH_LIBRARIES}
                                     Catch2::Catch2WithMain)
target_include_directories(test_precision PRIVATE
                                          operator
                                          util)

