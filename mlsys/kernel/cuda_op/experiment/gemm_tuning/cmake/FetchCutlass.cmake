include(FetchContent)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/emmett2020/cutlass.git
  GIT_TAG study-cutlass
)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(cutlass)
