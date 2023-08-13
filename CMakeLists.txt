cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

project(main)
add_executable(welcome main.cpp)

# Autodetect platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	message(STATUS "Configuring for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	message(STATUS "Configuring for macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	message(STATUS "Configuring for Windows")
else()
	message(STATUS "Configuring on/for ${CMAKE_SYSTEM_NAME}")
endif()

# Autodetect complier
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "using Clang compiler")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message(STATUS "using AppleClang compiler")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "using GCC compiler")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "using MSVC compiler")
else()
    message(STATUS "!unknow compiler. ${CMAKE_CXX_COMPILER_ID}")
endif()

# TODO: autodetect specific dependency exists then set $ENV_HAS_MYSQL to true
# TODO: default open thirdparties building.
option(ENV_HAS_MYSQL            "the build envrionment has mysql dependency"          OFF)
option(ENV_HAS_BOOST            "the build envrionment has boost dependency"          OFF)
option(ENV_HAS_FMT              "the build envrionment has fmt dependency"            OFF)
option(ENV_HAS_SPDLOG           "the build envrionment has spdlog dependency"         OFF)
option(ENV_HAS_NLOHMANN-JSON    "the build envrionment has nlohmann-json dependency"  OFF)

option(USE_XCODE            "use xcode to build projects"   OFF)

option(BUILD_DEMOS          "build utility demos"           OFF)
option(BUILD_LEETCODE       "build leetcode solutions"      OFF)
option(BUILD_PIECES         "build knowledge pieces"        OFF)
option(BUILD_PRESSURE       "build pressure"                OFF)
option(BUILD_TOOLS          "build tools"                   OFF)
option(BUILD_TUTORIAL       "build tutorial"                OFF)

if (BUILD_DEMOS)
        message(STATUS "Building demos. It may take several minutes......")
        add_subdirectory(demos)
else()
        message(STATUS "Building demos off. You may use -DBUILD_DEMOS=ON to open.")
endif()

if (BUILD_LEETCODE)
        message(STATUS "Building leetcode solutions. It may take several minutes......")
        add_subdirectory(leetcode)
else()
        message(STATUS "Building leetcode solutions off. You may use -DBUILD_LEETCODE=ON to open.")
endif()

if (BUILD_PIECES)
        message(STATUS "Building pieces. It may take several minutes......")
        add_subdirectory(pieces)
else()
        message(STATUS "Building pieces off. You may use -DBUILD_PIECES=ON to open.")
endif()

if (BUILD_TOOLS)
        message(STATUS "Building utility tools. It may take several minutes......")
        add_subdirectory(tools)
else()
        message(STATUS "Building utility tools off. You may use -DBUILD_TOOLS=ON to open.")
endif()

if (BUILD_TUTORIAL)
        message(STATUS "Building tutorial. It may take several minutes......")
        add_subdirectory(tutorial)
else()
        message(STATUS "Building tutorial off. You may use -DBUILD_TUTORIAL=ON to open.")
endif()

if (BUILD_BENCH)
        message(STATUS "Building benchmark. It may take several minutes......")
        add_subdirectory(bench)
else()
        message(STATUS "Building benchmark off. You may use -DBUILD_BENCH=ON to open.")
endif()

option(BUILD_THIRDPARTIES   "Build with thirdparties which some libraries needed."  ON)
if (BUILD_THIRDPARTIES)
        message(STATUS "Building thirdparties default open. You may use -DBUILD_THIRDPARTIES=OFF to close it.")
        add_subdirectory(thirdparties/nanobench)

        message(STATUS "google benchmark needs googletest. The library will automatically download and build any required dependencies.")
        message(STATUS "If you do not wish to build and run the tests, add -DBENCHMARK_ENABLE_GTEST_TESTS=OFF to close it")
        set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
        add_subdirectory(thirdparties/benchmark)
endif()


