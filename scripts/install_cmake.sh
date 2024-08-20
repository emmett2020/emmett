#!/bin/bash

set -e

CMAKE_VERSION=3.29.2
CMAKE_INSTALL_PATH="/usr/local"
CMAKE_LINK="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh"

function install_cmake() {
  local tmp_path="${HOME}/tmp"
  [[ -d "${tmp_path}" ]] && rm -rf "${tmp_path}"
  mkdir -p "${tmp_path}"

  echo "  Installing cmake ${CMAKE_VERSION} (needs sudo permission)"
  echo "  Link: ${CMAKE_LINK}"
  echo "  CMake will be installed into: ${CMAKE_INSTALL_PATH}"
  echo -e "  ......\n\n"

  wget ${CMAKE_LINK} -O "${tmp_path}/cmake_install.sh"
  sudo bash "${tmp_path}/cmake_install.sh" --skip-license --prefix="${CMAKE_INSTALL_PATH}"

  rm -rf "${tmp_path}"
  cmake --version
}

install_cmake
