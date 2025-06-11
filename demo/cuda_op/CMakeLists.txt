cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE op_files
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
pybind11_add_module(${SKBUILD_PROJECT_NAME} ${op_files})
target_include_directories(${SKBUILD_PROJECT_NAME}
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/operator
                           ${pybind11_INCLUDE_DIRS})
target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
                                               -Wno-c++23-extensions)
target_compile_features(${SKBUILD_PROJECT_NAME} PRIVATE
                                                cxx_std_23)
install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
