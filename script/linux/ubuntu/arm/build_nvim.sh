#!/bin/bash
: << 'COMMENT'
[DOING] Build neovim from source

|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |              No               |
|------------------------------|-------------------------------|
|          dependencies        |              No               |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

temp_dir=$(mktemp -d)
echo ${temp_dir}
# trap "rm -rf ${temp_dir}" EXIT
pushd ${temp_dir} &> /dev/null

git clone --recursive https://github.com/neovim/neovim.git
cd neovim

version="0.10.3"
git checkout "v${version}"

make CMAKE_BUILD_TYPE=RelWithDebInfo CMAKE_INSTALL_PREFIX=${temp_dir}/nvim
make install

popd &> /dev/null

