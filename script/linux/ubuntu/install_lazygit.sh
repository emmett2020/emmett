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

arch=$(uname -m)
[[ "${arch}" == "aarch64" ]] && arch="arm64"

version="0.45.0"
lazygit_link="https://github.com/jesseduffield/lazygit/releases/download/v${version}/lazygit_${version}_Linux_${arch}.tar.gz"
lazygit_dir="/usr/local/bin"

temp_dir=$(mktemp -d)
trap "rm -rf ${temp_dir}" EXIT

pushd ${temp_dir} &> /dev/null
wget "${lazygit_link}" -O lazygit.tar.gz
tar xf lazygit.tar.gz
sudo install lazygit ${lazygit_dir}
popd

lazygit --version
