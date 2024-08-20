#!/bin/bash
set -e

CUR_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
NEOVIM_VERSION="0.10.1"
NEOVIM_INSTALL_PATH="${HOME}/.neovim/"
NEOVIM_LINK="https://github.com/neovim/neovim/releases/download/v${NEOVIM_VERSION}/nvim-linux64.tar.gz"
NEOVIM_CONFIG_PATH="${HOME}/.config/nvim"

function get_neovim_version() {
	INSTALLED_VERSION=$(nvim --version | grep NVIM | awk -Fv '{print $2}')
	return $?
}

function install_neovim() {
    local tmp_path="${HOME}/.tmp_install"
    [[ -d "${tmp_path}" ]] && rm -rf "${tmp_path}"
    [[ -d "${NEOVIM_INSTALL_PATH}" ]] && rm -rf "${NEOVIM_INSTALL_PATH}"
    mkdir -p "${tmp_path}"
    mkdir -p "${NEOVIM_INSTALL_PATH}"

    echo "  Installing neovim ${NEOVIM_VERSION}"
    echo "  Link: ${NEOVIM_LINK}"
    echo "  Neovim will be installed into: ${NEOVIM_INSTALL_PATH}"
    echo "  Neovim configs will be installed into: ${NEOVIM_CONFIG_PATH}"
    echo -e "  ......\n\n"

    wget ${NEOVIM_LINK} -O "${tmp_path}/nvim-linux.tar.gz"

    local unzip_path="${tmp_path}/neovim"
    mkdir -p "${unzip_path}"
    tar -xzf "${tmp_path}/nvim-linux.tar.gz" -C ${unzip_path}
    mv "${unzip_path}/nvim-linux64/bin"   "${NEOVIM_INSTALL_PATH}/bin"
    mv "${unzip_path}/nvim-linux64/lib"   "${NEOVIM_INSTALL_PATH}/lib"
    mv "${unzip_path}/nvim-linux64/share" "${NEOVIM_INSTALL_PATH}/share"

    local emmett_path="${CUR_SCRIPT_DIR}/.."
    local default_cfg="${emmett_path}/configs/nvim"
 
    if [[ ! -d "${default_cfg}" ]]; then
      echo "  Cann't find nvim/ in emmett repo."
      echo "  Path of nvim: ${default_cfg} "
      exit 1
    fi

    [[ -d "${NEOVIM_CONFIG_PATH}" ]] && rm -r ${NEOVIM_CONFIG_PATH}
    cp -raf "${default_cfg}" "${NEOVIM_CONFIG_PATH}/"

    rm -rf ${tmp_path}
    ${NEOVIM_INSTALL_PATH}/bin/nvim --version

    echo -e "  Neovim installed successfully.\n"
    echo '  You should add '${HOME}/.neovim/bin' into $PATH in your .zshrc and enable it:'
    echo 'export PATH="${HOME}/.neovim/bin:$PATH"'
    echo 'source "${HOME}/.zshrc"'
    return 0
}

install_neovim
