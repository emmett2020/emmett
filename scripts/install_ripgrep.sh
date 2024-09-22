#!/bin/bash
set -e

ripgrep_version="14.1.0"
link_ripgrep="https://github.com/BurntSushi/ripgrep/releases/download/${ripgrep_version}/ripgrep_${ripgrep_version}-1_amd64.deb"

function print_hint() {
  echo -e "  ......\n\n"
  echo "  ripgrep download Link: ${link_ripgrep}"
}

function install_ripgrep() {
	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  wget ${link_ripgrep} -O "${tmp}/ripgrep.deb"
  sudo dpkg -i "${tmp}/ripgrep.deb"
  rm -rf ${tmp}
	rg --version
}

install_ripgrep
print_hint
