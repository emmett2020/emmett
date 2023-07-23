message (STATUS "building demos: multithreads_read_same_file")

include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_libraries(/opt/homebrew/Cellar/boost/1.75.0_2/lib)

add_executable(multithreads_read_same_file multithreads_read_same_file.cpp)