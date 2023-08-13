message (STATUS "Building perfomance: std_isxxx")
set(LIBS benchmark::benchmark)

add_executable(std_isxxx std_isxxx.cpp)
target_link_libraries(std_isxxx ${LIBS})