include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0
)
message(STATUS "Downloading Catch2")
FetchContent_MakeAvailable(Catch2)
