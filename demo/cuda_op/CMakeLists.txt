cmake_minimum_required(VERSION 3.26)
project(cuda_op LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

find_package(CUDAToolkit REQUIRED)

# include(cmake/FetchPybind11.cmake)
# set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE cuda_op_files
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
pybind11_add_module(cuda_op ${cuda_op_files})
# install(TARGETS cuda_op DESTINATION ./build)

set_target_properties(cuda_op PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_op PRIVATE
                                   ${CMAKE_CURRENT_SOURCE_DIR}/operator)
target_compile_options(cuda_op PRIVATE
                               -Wno-c++23-extensions)
target_compile_features(cuda_op PRIVATE
                                cxx_std_23)
