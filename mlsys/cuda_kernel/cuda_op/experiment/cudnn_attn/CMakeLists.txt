cmake_minimum_required(VERSION 3.26)
project(experiment LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

find_package(CUDAToolkit REQUIRED)

set(target_name cuda_exe)
file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
add_executable(${target_name} ${files})
target_compile_options(${target_name} PRIVATE -Wno-c++23-extensions)
target_compile_features(${target_name} PRIVATE cxx_std_23)

# TODO: This is hard code.
set(CUDNN_PATH "/home/emmett/miniconda3/envs/cuda_op/")
target_include_directories(${target_name} PRIVATE "${CUDNN_PATH}/include")
target_link_libraries(${target_name} PRIVATE "${CUDNN_PATH}/lib/libcudnn.so")
