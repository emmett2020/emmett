#!/bin/bash

set -e
echo "Installing cmake"

CMAKE_VERSION=3.29.2
CMAKE_INSTALL_PATH="/usr/local"
CMAKE_TEMP_PATH="/tmp/cmake_${CMAKE_VERSION}.sh"

wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" -O ${CMAKE_TEMP_PATH}
sudo bash ${CMAKE_TEMP_PATH} --skip-license --prefix="${CMAKE_INSTALL_PATH}" 
