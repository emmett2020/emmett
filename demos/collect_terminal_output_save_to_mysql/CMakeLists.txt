message (STATUS "building demos: collect_terminal_output_save_to_mysql")

# boost
include_directories(/opt/homebrew/Cellar/boost/1.75.0_2/include)
link_directories(/opt/homebrew/Cellar/boost/1.75.0_2/lib)

# spdlog
include_directories(/opt/homebrew/Cellar/spdlog/1.8.2/include)
link_libraries(/opt/homebrew/Cellar/spdlog/1.8.2/lib)


# nlohmann-json
include_directories(/opt/homebrew/Cellar/nlohmann-json/3.9.1_1/include)
link_libraries(/opt/homebrew/Cellar/nlohmann-json/3.9.1_1/lib)

# fmt
include_directories(/opt/homebrew/Cellar/fmt/7.1.3/include)
link_libraries(/opt/homebrew/Cellar/fmt/7.1.3/lib)

set(SRC_LIST vars.h exec_cmd.h recv_cmd.h print_cmd_res.h)

add_executable(collect_terminal_output_save_to_mysql main.cpp ${SRC_LIST})