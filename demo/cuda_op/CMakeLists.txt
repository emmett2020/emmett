cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)

find_package(CUDAToolkit REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE cuda_op_files
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
python_add_library(${SKBUILD_PROJECT_NAME} MODULE ${cuda_op_files} WITH_SOABI)
# pybind11_add_module(${SKBUILD_PROJECT_NAME} ${cuda_op_files})

set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/operator)
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE pybind11::headers)

target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
                                               -Wno-c++23-extensions)
target_compile_features(${SKBUILD_PROJECT_NAME} PRIVATE
                                                cxx_std_23)
install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
