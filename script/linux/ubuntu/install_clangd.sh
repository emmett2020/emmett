#!/bin/bash
cat << END
Install pre-built clangd
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
END
set -euo pipefail

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

pushd "${temp_dir}" &> /dev/null

version="19.1.7"
arch=$(uname -m)
[[ "${arch}" == 'aarch64' ]] && arch=arm
[[ "${arch}" == 'x86_64' ]] && arch=amd

link="https://github.com/emmett2020/llvm-prebuilt-binary/releases/download/v${version}-${arch}/llvm-${arch}.tar.gz"

wget "${link}" -O llvm.tar.gz
tar xf llvm.tar.gz
sudo mv llvm/bin/* /usr/local/bin/
sudo mv llvm/lib/clang/* /usr/local/lib/clang/
popd

clangd --version
