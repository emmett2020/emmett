message (STATUS "Building turotial: design pattern")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tutorial/design_pattern/)
add_executable(01_simple_factory 01_simple_factory.cpp)
add_executable(02_factory_method 02_factory_method.cpp)
