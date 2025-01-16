#!/bin/bash

set -e

nvim_version="0.10.1"
link_ubuntu_nvim="https://github.com/neovim/neovim/releases/download/v${nvim_version}/nvim-linux64.tar.gz"
dir_nvim="${HOME}/.neovim/"
dir_nvim_config="${HOME}/.config/nvim"

cur_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
dir_emmett="${cur_dir}/.."
dir_emmett_nvim_config="${dir_emmett}/configs/nvim"
dir_temp="${HOME}/.tmp_install"

function check_nvim_config() {
    if [[ ! -d "${dir_emmett_nvim_config}" ]]; then
      echo "  Cann't find nvim/ in emmett repo. Search path: ${dir_emmett_nvim_config}"
      exit 1
    fi
    if [[ -d "${dir_nvim_config}" ]]; then
      echo "  Remove or save ${dir_nvim_config} first."
      exit 1
    fi

    mkdir -p "${dir_nvim_config}"
    cp -r "${dir_emmett_nvim_config}/"* "${dir_nvim_config}/"
}


function print_hint() {
    echo "  Neovim installed path: ${dir_nvim}"
    echo "  Neovim configs   path: ${dir_nvim_config}"
    echo "  Neovim ${nvim_version} installed successfully."

    echo '  You should add '${HOME}/.neovim/bin' into $PATH in your .zshrc and enable it.'
}


function install_neovim_ubuntu() {
    wget ${link_ubuntu_nvim} -O "${dir_temp}/nvim.tar.gz"

    local unzip_path="${dir_temp}/neovim"
    mkdir -p "${unzip_path}"
    tar -xzf "${dir_temp}/nvim.tar.gz" -C ${unzip_path}

    mv "${unzip_path}/nvim-linux64/bin"   "${dir_nvim}/bin"
    mv "${unzip_path}/nvim-linux64/lib"   "${dir_nvim}/lib"
    mv "${unzip_path}/nvim-linux64/share" "${dir_nvim}/share"
 
    rm -rf ${dir_temp}
    ${dir_nvim}/bin/nvim --version
}

function install_neovim_macos() {
  brew install neovim
  nvim --version
}

##############################################
#               entrypoint
##############################################

check_nvim_config
source "${cur_dir}/detect_os.sh"

[[ -d "${dir_temp}" ]] && rm -rf "${dir_temp}"
[[ -d "${dir_nvim}" ]] && rm -rf "${dir_nvim}"
mkdir -p "${dir_temp}"
mkdir -p "${dir_nvim}"

if [[ "${OS}" == "Ubuntu" ]]; then
  install_neovim_ubuntu
elif [[ "${OS}" == "MacOS" ]]; then
  install_neovim_macos
else
  exit 1
fi

rm -rf ${dir_temp}
print_hint
