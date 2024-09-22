#!/bin/bash

set -e

cmake_version=3.29.2
dir_cmake="/usr/local"
link_cmake="https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-linux-x86_64.sh"


function print_hint() {
  echo -e "  ......\n\n"
  echo "  Installed cmake ${cmake_version}"
  echo "  cmake download  link: ${link_cmake}"
  echo "  CMake installed path: ${dir_cmake}"
}

function install_cmake() {
  local tmp_path="${HOME}/.tmp_install"
  [[ -d "${tmp_path}" ]] && rm -rf "${tmp_path}"
  mkdir -p "${tmp_path}"

  wget ${link_cmake} -O "${tmp_path}/cmake_install.sh"
  sudo bash "${tmp_path}/cmake_install.sh" --skip-license --prefix="${dir_cmake}"

  rm -rf "${tmp_path}"
  cmake --version
}

install_cmake
print_hint
