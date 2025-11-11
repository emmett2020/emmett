cmake_minimum_required(VERSION 3.26)
project(experiment LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

set(target_name cuda_exe)
file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
add_executable(${target_name} ${files})
target_compile_options(${target_name} PRIVATE -Wno-c++23-extensions)
target_compile_features(${target_name} PRIVATE cxx_std_23)
