#!/bin/bash
set -e

CPPDBG_VERSION="1.21.6"
CPPDBG_LINK="https://github.com/microsoft/vscode-cpptools/releases/download/v${CPPDBG_VERSION}/cpptools-linux.vsix"
CPPDBG_INSTALL_PATH="$HOME/.local/share/nvim/cppdbg/"

function exist_cppdbg() {
	# Just use path of OpenDebugAD7 to check wheter cppdbg exists.
	local p="$CPPDBG_INSTALL_PATH/extension/debugAdapters/bin/OpenDebugAD7"
	if [[ -e $p ]]; then
		return 0
	else
		return 1
	fi
}

function get_cppdbg_version() {
  echo "We don't have a good way to get cppdbg version."
}

function install_cppdbg() {
	local vsix_path="/tmp/cpptools-linux.vsix"
	local unzip_path="/tmp/cppdbg/"
  echo " Installing cppdbg ${CPPDBG_VERSION} ......"
  echo " Link: ${CPPDBG_LINK}"
  wget  ${CPPDBG_LINK} -O ${vsix_path}
  unzip ${vsix_path} -d ${unzip_path}
  [[ -d ${CPPDBG_INSTALL_PATH} ]] && rm ${CPPDBG_INSTALL_PATH}
  mkdir -p ${CPPDBG_INSTALL_PATH}
  mv "${unzip_path}/extension" "${CPPDBG_INSTALL_PATH}"
  chmod +x "${CPPDBG_INSTALL_PATH}/extension/debugAdapters/bin/OpenDebugAD7"
  sudo rm ${vsix_path}
  sudo rm -r ${unzip_path}
	echo "Installed cppdbg $version."
}
