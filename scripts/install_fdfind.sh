#!/bin/bash
set -e

fdfind_version="10.1.0"
link_fdfind="https://github.com/sharkdp/fd/releases/download/v${fdfind_version}/fd_${fdfind_version}_amd64.deb"

function print_hint() {
  echo -e "  ......\n\n"
  echo "  Installed fdfind ${fdfind_version}"
  echo "  fdfind download link: ${link_fdfind}"
}

function install_fdfind() {
	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  wget ${link_fdfind} -O "${tmp}/fdfind.deb"
  sudo dpkg -i "${tmp}/fdfind.deb"
  rm -r ${tmp}
	fd --version
}

install_fdfind
print_hint

