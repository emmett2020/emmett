#!/bin/bash
set -e

CODELLDB_VERSION="1.10.0"
CODELLDB_INSTALL_PATH="$HOME/.local/share/nvim/codelldb/"
CODELLDB_LINK="https://github.com/vadimcn/codelldb/releases/download/v${CODELLDB_VERSION}/codelldb-x86_64-linux.vsix"

function exist_codelldb() {
	local p="$CODELLDB_INSTALL_PATH/extension/adapter/codelldb"
	if [[ -e $p ]]; then
		return 0
	else
		return 1
	fi
}

function install_codelldb() {
  echo "  codelldb Link: ${CODELLDB_LINK}"
  echo "  codelldb will be installed into: ${CODELLDB_INSTALL_PATH}"
  echo -e "  ......\n\n"

	local tmp_path="${HOME}/.tmp_install"
	local unzip_path="${tmp_path}/codelldb/"

  [[ -d "${CODELLDB_INSTALL_PATH}" ]] && rm -r "${CODELLDB_INSTALL_PATH}"
  mkdir -p "${CODELLDB_INSTALL_PATH}"
  mkdir -p "${unzip_path}"

  wget ${CODELLDB_LINK} -O "${tmp_path}/codelldb.vsix"
  unzip -q "${tmp_path}/codelldb.vsix" -d ${unzip_path}
  mv "${unzip_path}/extension" ${CODELLDB_INSTALL_PATH}
  chmod +x "${CODELLDB_INSTALL_PATH}/extension/adapter/codelldb"

  rm -r ${tmp_path}
  echo "  Installed codelldb: ${CODELLDB_VERSION}"
}

install_codelldb
