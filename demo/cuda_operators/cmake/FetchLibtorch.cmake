include(FetchContent)

# https://download.pytorch.org/libtorch/cpu
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip")

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
)

FetchContent_MakeAvailable(libtorch)
list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
