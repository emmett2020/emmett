include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13
)
message(STATUS "Downloading pybind11")
FetchContent_MakeAvailable(pybind11)
