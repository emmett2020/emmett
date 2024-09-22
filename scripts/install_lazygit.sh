#!/bin/bash
set -e

lazygit_version="0.43.1"
link_lazygit="https://github.com/jesseduffield/lazygit/releases/download/v${lazygit_version}/lazygit_${lazygit_version}_Linux_x86_64.tar.gz"
dir_lazygit="/usr/local/bin"


function print_hint() {
  echo -e "  ......\n\n"
  echo "  Lazygit download  link: ${link_lazygit}"
  echo "  Lazygit installed path: ${dir_lazygit}"
}

function install_lazygit() {
	local tmp="${HOME}/.tmp_install"
	local tmp_gz="${tmp}/lazygit.tar.gz"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p ${tmp}

  wget "${link_lazygit}" -O "${tmp_gz}"
  tar xf ${tmp_gz} -C ${tmp}
  sudo install "${tmp}"/lazygit ${dir_lazygit}
  rm -rf ${tmp}
	lazygit --version
}

install_lazygit
print_hint
