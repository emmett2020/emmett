#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
|          dependencies        |              No              |
|------------------------------|------------------------------|
END
set -euo pipefail

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

version="14.1.1"
arch=$(uname -m)
if [[ "${arch}" == 'aarch64' ]]; then
  ripgrep_link="https://github.com/BurntSushi/ripgrep/releases/download/${version}/ripgrep-${version}-aarch64-unknown-linux-gnu.tar.gz"

  pushd "${temp_dir}" &> /dev/null
  wget ${ripgrep_link} -O ripgrep.tar.gz
  tar xf ripgrep.tar.gz
  cd "ripgrep-${version}-aarch64-unknown-linux-gnu"
  sudo mv rg /usr/local/bin
  popd
else
  ripgrep_link="https://github.com/BurntSushi/ripgrep/releases/download/${version}/ripgrep_${version}-1_amd64.deb"
  wget ${ripgrep_link} -O "${temp_dir}/ripgrep.deb"
  sudo dpkg -i "${temp_dir}/ripgrep.deb"
fi

rg --version
