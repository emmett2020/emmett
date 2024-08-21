#!/bin/bash
set -e

CPPDBG_VERSION="1.21.6"
CPPDBG_LINK="https://github.com/microsoft/vscode-cpptools/releases/tag/v${CPPDBG_VERSION}/cpptools-linux-x64.vsix"
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

function install_cppdbg() {
  echo "  Link: ${CPPDBG_LINK}"
  echo "  cppdbg will be installed into: ${CPPDBG_INSTALL_PATH}"
  echo -e "  ......\n\n"

	local tmp_path="${HOME}/.tmp_install"
	local unzip_path="${tmp_path}/cppdbg/"

  [[ -d "${CPPDBG_INSTALL_PATH}" ]] && rm -r "${CPPDBG_INSTALL_PATH}"
  mkdir -p "${CPPDBG_INSTALL_PATH}"
  mkdir -p "${unzip_path}"

  wget ${CPPDBG_LINK} -O "${tmp_path}/cpptools.vsix"
  unzip -q "${tmp_path}/cpptools.vsix" -d ${unzip_path}
  mv "${unzip_path}/extension" "${CPPDBG_INSTALL_PATH}"
  chmod +x "${CPPDBG_INSTALL_PATH}/extension/debugAdapters/bin/OpenDebugAD7"

  rm -r ${tmp_path}
  echo "  Installed cppdbg: ${CPPDBG_VERSION}"
}

# TODO: ERROR
# install_cppdbg
