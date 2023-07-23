set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tools/)

if (ENV_HAS_MYSQL AND ENV_HAS_FMT)
  add_subdirectory(add_data_to_mysql)
endif()

add_subdirectory(collect_terminal_output)