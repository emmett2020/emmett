#!/bin/bash
set -e

LAZYGIT_VERSION="0.43.1"
LAZYGIT_LINK="https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
LAZYGIT_INSTALL_PATH="/usr/local/bin"

function get_lazygit_version() {
	local installed_version=$(lazygit --version | awk -F, '{print $4}' | awk -F= '{print $2}')
  echo ${installed_version}
}

function install_lazygit() {
  echo "  Installing lazygit ${LAZYGIT_VERSION} (needs sudo permission) ......"
  echo "  Lazygit Link: ${LAZYGIT_LINK}"
  echo "  Lazygit install path: ${LAZYGIT_INSTALL_PATH}"
  echo -e "  ......\n\n"

	local tmp="${HOME}/.tmp_install"
	local tmp_gz="${tmp}/lazygit.tar.gz"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p ${tmp}

  wget "${LAZYGIT_LINK}" -O "${tmp_gz}"
  tar xf ${tmp_gz} -C ${tmp}
  sudo install "${tmp}"/lazygit ${LAZYGIT_INSTALL_PATH}
  rm -rf ${tmp}
	lazygit --version
}

install_lazygit
