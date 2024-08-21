#!/bin/bash
set -e

FDFIND_VERSION="10.1.0"
FDFIND_LINK="https://github.com/sharkdp/fd/releases/download/v${FDFIND_VERSION}/fd_${FDFIND_VERSION}_amd64.deb"

function get_fdfind_version() {
	INSTALLED_VERSION=$(fd --version | grep fd | awk '{print $2}')
	return $?
}

function install_fdfind() {
  echo "  Installing fdfind ${FDFIND_VERSION} (needs sudo permission) ......"
  echo "  fdfind Link: ${FDFIND_LINK}"
  echo -e "  ......\n\n"

	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  wget ${FDFIND_LINK} -O "${tmp}/fdfind.deb"
  sudo dpkg -i "${tmp}/fdfind.deb"
  rm -r ${tmp}
	fd --version
}

install_fdfind

