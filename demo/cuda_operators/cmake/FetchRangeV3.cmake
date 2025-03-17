include(FetchContent)

FetchContent_Declare(
  ranges-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  GIT_TAG ca1388f
)

message(STATUS "Downloading ranges-v3")
FetchContent_MakeAvailable(ranges-v3)
