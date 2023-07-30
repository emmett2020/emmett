set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/pieces/)

add_subdirectory(simply_use_coroutine)
add_subdirectory(simply_use_sigbus)

if (ENV_HAS_BOOST)
  add_subdirectory(simple_use_boost_message_queue)
endif()

add_subdirectory(string_view_or_string)

