include(FetchContent)

FetchContent_Declare(
  proxy
  GIT_REPOSITORY https://github.com/microsoft/proxy.git
  GIT_TAG 3.3.0
)

message(STATUS "Downloading proxy")
FetchContent_MakeAvailable(proxy)
