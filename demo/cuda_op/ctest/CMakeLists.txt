cmake_minimum_required(VERSION 3.26)
project(test_cuda_op LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90)

find_package(CUDAToolkit REQUIRED)

function(create_exe cpp_file)
    get_filename_component(target_name ${cpp_file} NAME_WE)
    add_executable(${target_name} ${cpp_file})
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_include_directories(${target_name}
                              PRIVATE
                              ${CMAKE_CURRENT_SOURCE_DIR}/../operator)
    target_compile_options(${target_name} PRIVATE -Wno-c++23-extensions)
    target_compile_features(${target_name} PRIVATE cxx_std_23)
    message(STATUS "Added executable target: ${target_name} from ${cpp_file}")
endfunction()

macro(process_files_in_directory)
    file(GLOB_RECURSE cu_files
        LIST_DIRECTORIES false
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cu"
    )
    foreach(cu_file ${cu_files})
        create_exe(${cu_file})
    endforeach()
endmacro()

process_files_in_directory()
