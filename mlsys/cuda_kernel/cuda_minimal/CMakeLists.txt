# https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd
cmake_minimum_required(VERSION 3.26)
project(cuda-minimal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
# NOTE: If you have multiple CUDA version installed, plz carefully read above
# links first to use correct version.
find_package(CUDAToolkit REQUIRED)

set(target_name cuda_exe)
add_executable(${target_name} main.cu)
target_compile_options(${target_name} PRIVATE -Wno-c++23-extensions)
target_compile_features(${target_name} PRIVATE cxx_std_23)

