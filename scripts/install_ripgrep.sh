#!/bin/bash

RIPGREP_VERSION="14.1.0"
RIPGREP_LINK="https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/ripgrep_${RIPGREP_VERSION}_amd64.deb"

# Get ripgrep version installed in this machine.
function get_ripgrep_version() {
	INSTALLED_VERSION=$(rg --version | grep ripgrep | awk '{print $2}')
	return $?
}

function install_ripgrep() {
	local tmp="/tmp/ripgrep.deb"
  echo " Installing ripgrep ${RIPGREP_VERSION} ......"
  echo " Link: ${RIPGREP_LINK}"
  wget ${RIPGREP_LINK} -O ${tmp}
  sudo dpkg -i ${tmp}
  rm ${tmp}
	rg --version
}
