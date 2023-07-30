message (STATUS "Building pieces: string_view_or_string")
set(LIBS benchmark::benchmark)

add_executable(memory_allocation memory_allocation.cpp)
target_link_libraries(memory_allocation ${LIBS})

add_executable(interaction_with_other_components interaction_with_other_components.cpp)
target_link_libraries(interaction_with_other_components ${LIBS})