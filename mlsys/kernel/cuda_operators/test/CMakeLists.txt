set(ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")

file(GLOB_RECURSE test_op_files "*.cpp" "*.cu")
add_executable(test_op ${test_op_files})

set_target_properties(test_op PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_op PRIVATE
                              ${TORCH_LIBRARIES}
                              msft_proxy
                              range-v3
                              util
                              accuracy
                              Catch2::Catch2WithMain)

target_include_directories(test_op PRIVATE
                                   ${CMAKE_CURRENT_SOURCE_DIR}/common
                                   ${TORCH_INCLUDE_DIRS}
                                   ${ROOT_PATH}/operator
                                   ${ROOT_PATH}/util
                                   ${ROOT_PATH}/accuracy)

target_compile_definitions(Catch2 PUBLIC
                                  CATCH_CONFIG_PREFIX_ALL)
target_compile_options(test_op PRIVATE
                               -Wno-c++23-extensions)
target_compile_features(test_op PRIVATE
                                cxx_std_23)
