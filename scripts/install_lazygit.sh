#!/bin/bash
set -e

LAZYGIT_VERSION="0.43.1"
LAZYGIT_LINK="https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"

function get_lazygit_version() {
	local installed_version=$(lazygit --version | awk -F, '{print $4}' | awk -F= '{print $2}')
  echo ${installed_version}
}

function install_lazygit() {
	local tmp_gz="/tmp/lazygit.tar.gz"
	local tmp="/tmp/lazygit"
	local install_path="/usr/local/bin"

  echo " Installing lazygit ${LAZYGIT_VERSION} ......"
  echo " Link: ${LAZYGIT_LINK}"
  wget ${LAZYGIT_LINK} -O ${tmp_gz}
  sudo mkdir ${tmp}
  sudo tar xf ${tmp_gz} -C ${tmp}
  sudo install "${tmp}"/lazygit ${install_path}
  sudo rm -rf $tmp
  sudo rm $tmp_gz
	lazygit --version
}
