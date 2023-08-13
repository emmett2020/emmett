message (STATUS "Building perfomance: std_isdigit")
set(LIBS benchmark::benchmark)

add_executable(std_isdigit std_isdigit.cpp)
target_link_libraries(std_isdigit ${LIBS})