#!/bin/bash
: << 'COMMENT'
Build neovim from source

|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |            Yes                |
|------------------------------|-------------------------------|
|          dependencies        |              No               |
|------------------------------|-------------------------------|
|          args                |           install_dir         |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

[[ "$#" -lt 1 ]] && echo "Must provide neovim installation directory" && exit 1

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT
pushd "${temp_dir}"

sudo apt install -y ninja-build gettext cmake curl build-essential
git clone --recursive https://github.com/neovim/neovim.git
cd neovim

version="0.10.3"
git checkout "v${version}"

echo "::group:: build nvim"
make -j "$(nproc)" CMAKE_BUILD_TYPE=RelWithDebInfo CMAKE_INSTALL_PREFIX="$1"
make -j "$(nproc)" install
echo "::endgroup::"

popd &> /dev/null
