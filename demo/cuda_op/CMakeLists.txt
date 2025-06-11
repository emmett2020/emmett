cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Build CUDA shared library
file(GLOB_RECURSE cuda_files "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
set_source_files_properties(${cuda_files} PROPERTIES LANGUAGE CUDA)
add_library(cuda_code SHARED ${cuda_files})
set_target_properties(cuda_code PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_code PRIVATE
                                     ${CMAKE_CURRENT_SOURCE_DIR}/operator
                                     ${pybind11_INCLUDE_DIRS})
target_compile_options(cuda_code PRIVATE -Wno-c++23-extensions)
target_compile_features(cuda_code PRIVATE cxx_std_23)
install(TARGETS cuda_code DESTINATION ${SKBUILD_PROJECT_NAME})

# Build CPP shared library
file(GLOB_RECURSE cpp_files "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp")
pybind11_add_module(${SKBUILD_PROJECT_NAME} ${cpp_files})
target_include_directories(${SKBUILD_PROJECT_NAME}
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/operator
                           ${pybind11_INCLUDE_DIRS})
target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
                                               -Wno-c++23-extensions)
target_compile_features(${SKBUILD_PROJECT_NAME} PRIVATE
                                                cxx_std_23)
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE cuda_code)
install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
