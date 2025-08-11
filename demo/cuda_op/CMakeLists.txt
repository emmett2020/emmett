cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

file(GLOB_RECURSE cpp_files
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cpp")

file(GLOB_RECURSE cu_files
                  "${CMAKE_CURRENT_SOURCE_DIR}/operator/conv/*.cu")
# file(GLOB_RECURSE cu_files
#                   "${CMAKE_CURRENT_SOURCE_DIR}/operator/*.cu")
pybind11_add_module(${SKBUILD_PROJECT_NAME} ${cpp_files} ${cu_files})
target_include_directories(${SKBUILD_PROJECT_NAME}
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/operator
                           ${TORCH_INCLUDE_DIRS}
                           ${pybind11_INCLUDE_DIRS})

# INFO: for https://github.com/pytorch/pytorch/issues/108041
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE
                                              ${TORCH_PYTHON_LIBRARY}
                                              ${TORCH_LIBRARIES})
target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
                                               -lineinfo
                                               -Wno-c++23-extensions)
target_compile_options(${SKBUILD_PROJECT_NAME} PRIVATE
                                               $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas -dlcm=cg>)
target_compile_features(${SKBUILD_PROJECT_NAME} PRIVATE
                                                cxx_std_23)
install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
