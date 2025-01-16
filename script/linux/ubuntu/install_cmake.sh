#!/bin/bash
: << 'COMMENT'
|------------------------------|------------------------------|
|         🎃 item              |        👇 explanation        |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
|          dependencies        |              No              |
|------------------------------|------------------------------|
COMMENT

set -euo pipefail

cmake_version=3.29.2
cmake_link="https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-linux-x86_64.sh"
cmake_dir="/usr/local"

temp_dir=$(mktemp -d)
trap "rm -rf ${temp_dir}" EXIT

wget ${cmake_link} -O "${temp_dir}/cmake_install.sh"
sudo bash "${temp_dir}/cmake_install.sh" --skip-license --prefix="${cmake_dir}"
cmake --version
