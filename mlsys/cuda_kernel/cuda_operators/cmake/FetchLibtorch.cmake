include(FetchContent)

# https://pytorch.org/
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip")

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
)

FetchContent_MakeAvailable(libtorch)
message(STATUS "Downloading libtorch")
list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
