#!/bin/bash
set -e

FDFIND_VERSION="10.1.0"
FDFIND_LINK="https://github.com/sharkdp/fd/releases/download/v${FDFIND_VERSION}/fd_${FDFIND_VERSION}_amd64.deb"

function get_fdfind_version() {
	INSTALLED_VERSION=$(fd --version | grep fd | awk '{print $2}')
	return $?
}

function install_fdfind() {
	local tmp="/tmp/fdfind.deb"
  echo " Installing fdfind ${FDFIND_VERSION} ......"
  echo " Link: ${RIPGREP_LINK}"
  wget ${RIPGREP_LINK} -O ${tmp}
  sudo dpkg -i ${tmp}
  rm ${tmp}
	fd --version
}

