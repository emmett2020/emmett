#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
END
set -euo pipefail

arch=$(uname -m)
[[ "${arch}" == "x86_64" ]] && arch="amd64"
[[ "${arch}" == "aarch64" ]] && arch="arm64"

fdfind_version="10.2.0"
fdfind_link="https://github.com/sharkdp/fd/releases/download/v${fdfind_version}/fd_${fdfind_version}_${arch}.deb"

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

wget "${fdfind_link}" -O "${temp_dir}/fdfind.deb"
sudo dpkg -i "${temp_dir}/fdfind.deb"
fd --version
