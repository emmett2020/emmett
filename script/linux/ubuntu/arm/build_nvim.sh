#!/bin/bash
: << 'COMMENT'
[DOING] Build neovim from source

|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |            Not Sure           |
|------------------------------|-------------------------------|
|          dependencies        |              No               |
|------------------------------|-------------------------------|
|          args                |           install_dir         |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

[[ "$@" == "" ]] && echo "Must provide neovim installation dirctory" && exit 1

temp_dir=$(mktemp -d)
trap "rm -rf ${temp_dir}" EXIT
pushd ${temp_dir} &> /dev/null

git clone --recursive https://github.com/neovim/neovim.git
cd neovim

version="0.10.3"
git checkout "v${version}"

echo "::group:: build nvim"
make -j`nproc` CMAKE_BUILD_TYPE=RelWithDebInfo CMAKE_INSTALL_PREFIX="$@"
make -j`nproc` install
echo "::endgroup::"

popd &> /dev/null

