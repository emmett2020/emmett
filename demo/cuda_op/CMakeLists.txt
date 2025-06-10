cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
  set(DEFAULT_NVCC_PATH "/usr/local/cuda/bin/nvcc")
  message(STATUS "CMAKE_CUDA_COMPILER is not defined. Use default path: ${DEFAULT_NVCC_PATH}")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


message("XXX  ${pybind11_INCLUDE_DIRS}" )

# Build CUDA shared library
file(GLOB_RECURSE cuda_files "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
cuda_add_library(cuda_code SHARED ${cuda_files})
set_target_properties(cuda_code PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_code PRIVATE
                                     ${CMAKE_CURRENT_SOURCE_DIR}/operator
                                     ${pybind11_INCLUDE_DIRS})
target_compile_options(cuda_code PRIVATE -Wno-c++23-extensions)
target_compile_features(cuda_code PRIVATE cxx_std_23)
install(TARGETS cuda_code DESTINATION ${SKBUILD_PROJECT_NAME})

# Build CPP shared library
# file(GLOB_RECURSE cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp")
# pybind11_add_module(${SKBUILD_PROJECT_NAME} ${cpp_files})
# set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE
#                                                    ${CMAKE_CURRENT_SOURCE_DIR}/operator)
# target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
#                                                -Wno-c++23-extensions)
# target_compile_features(${SKBUILD_PROJECT_NAME} PRIVATE
#                                                 cxx_std_23)
# install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
