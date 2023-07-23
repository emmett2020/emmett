message (STATUS "Building pieces: simple_use_boost_message_queue")
include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_directories(/opt/homebrew/Cellar/boost/1.75.0_2/lib)
add_executable(simple_use_boost_message_queue simple_use_boost_message_queue.cpp)