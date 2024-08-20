#!/bin/bash
set -e

RIPGREP_VERSION="14.1.0"
RIPGREP_LINK="https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/ripgrep_${RIPGREP_VERSION}-1_amd64.deb"

# Get ripgrep version installed in this machine.
function get_ripgrep_version() {
	INSTALLED_VERSION=$(rg --version | grep ripgrep | awk '{print $2}')
	return $?
}

function install_ripgrep() {
  echo "  Installing ripgrep ${RIPGREP_VERSION} ......"
  echo "  Link: ${RIPGREP_LINK}"

	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  wget ${RIPGREP_LINK} -O "${tmp}/ripgrep.deb"
  sudo dpkg -i "${tmp}/ripgrep.deb"
  rm -rf ${tmp}
	rg --version
}

install_ripgrep
