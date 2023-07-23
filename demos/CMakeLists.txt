set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/demos/)

if (ENV_HAS_BOOST)
  add_subdirectory(multithreads_read_same_file)
endif()

if (USE_XCODE)
  add_subdirectory(calculator)
endif()

if ((ENV_HAS_BOOST) AND (ENV_HAS_FMT) AND (ENV_HAS_SPDLOG) AND (ENV_HAS_NLOHMANN-JSON) AND (ENV_HAS_MYSQL))
  add_subdirectory(collect_terminal_output_save_to_mysql)
endif()k