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

temp_dir=$(mktemp -d)
# trap "rm -rf ${temp_dir}" EXIT

clangd_version=llvmorg-19.1.7
clangd_link="https://github.com/llvm/llvm-project.git"
clangd_dir="/usr/local"

pushd ${temp_dir} &> /dev/null
sudo apt install ninja-build

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout "${clangd_version}"
git submodule update --init --recursive
mkdir build && cd build
cmake ../llvm/ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -GNinja
cmake --build . --target clangd -j`nproc`

popd &> /dev/null

clangd --version
