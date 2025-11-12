cmake_minimum_required(VERSION 3.26)
project(emmett LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Debug)

include(cmake/FetchCatch2.cmake)
include(cmake/FetchLibtorch.cmake)
include(cmake/FetchProxy.cmake)
include(cmake/FetchRangeV3.cmake)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

file(GLOB_RECURSE util_file "util/*.cpp")
add_library(util ${util_file})
target_link_libraries(util PRIVATE range-v3)
target_compile_features(util PRIVATE cxx_std_23)

file(GLOB_RECURSE accuracy_file "accuracy/*.cpp")
add_library(accuracy ${accuracy_file})
target_link_libraries(accuracy PRIVATE range-v3 ${TORCH_LIBRARIES})
target_compile_features(accuracy PRIVATE cxx_std_23)

set(BUILD_TESTING on)
if (BUILD_TESTING)
  add_subdirectory(test)
endif()
