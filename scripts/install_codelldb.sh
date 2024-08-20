#!/bin/bash

CODELLDB_VERSION="1.10.0"
CODELLDB_INSTALL_PATH="$HOME/.local/share/nvim/codelldb/"
CODELLDB_LINK="https://github.com/vadimcn/codelldb/releases/download/v${CODELLDB_LINK}/codelldb-x86_64-linux.vsix"

function exist_codelldb() {
	local p="$CODELLDB_INSTALL_PATH/extension/adapter/codelldb"
	if [[ -e $p ]]; then
		return 0
	else
		return 1
	fi
}

function install_codelldb() {
	local vsix_path="/tmp/codelldb.vsix"
	local unzip_path="/tmp/codelldb/"

  echo " Installing codelldb ${CODELLDB_VERSION} ......"
  echo " Link: ${CODELLDB_LINK}"
  wget ${CODELLDB_LINK} -O ${vsix_path}
  unzip ${vsix_path} -d ${unzip_path}
  mkdir -p $CODELLDB_INSTALL_PATH
  mv "${unzip_path}/extension" ${CODELLDB_INSTALL_PATH}
  chmod +x "${CODELLDB_INSTALL_PATH}/extension/adapter/codelldb"
  sudo rm ${vsix_path}
  sudo rm -r ${unzip_path}
	echo "Installed codelldb ${CODELLDB_VERSION}."
}
