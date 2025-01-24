#!/bin/bash
cat << END
Build clangd;clang from source
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
|          args                |           install_dir        |
|------------------------------|------------------------------|
END
set -euo pipefail

[[ $# -lt 1 ]] && echo "Must provide clangd installation directory" && exit 1
install_prefix="$1"

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

clangd_version=llvmorg-19.1.7

pushd "${temp_dir}" &> /dev/null
sudo apt install ninja-build

git clone --depth=1 --branch ${clangd_version} https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake ../llvm/ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -GNinja -DCMAKE_INSTALL_PREFIX="${install_prefix}"
ninja -j"$(nproc)" install

popd &> /dev/null

"${install_prefix}"/bin/clangd --version
