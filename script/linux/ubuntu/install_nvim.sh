#!/bin/bash
: << 'COMMENT'
|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |              No               |
|------------------------------|-------------------------------|
|          dependencies        |  ${emmett}/config/nvim/       |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

cur_dir=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)
emmett_root_path="${cur_dir}/../../.."

nvim_install_dir="${HOME}/.neovim/"
[[ -d "${nvim_install_dir}" ]] && rm -rf "${nvim_install_dir}"
mkdir -p "${nvim_install_dir}"

function install_neovim_x86_64() {
    # Download nvim pre-built binary tarball to local
    local version="0.10.3"
    local nvim_link="https://github.com/neovim/neovim/releases/download/v${version}/nvim-linux64.tar.gz"
    wget ${nvim_link} -O "${temp_dir}/nvim.tar.gz"

    # Unzip the tarball
    local unzip_path="${temp_dir}/neovim"
    mkdir -p "${unzip_path}"
    tar -xzf "${temp_dir}/nvim.tar.gz" -C "${unzip_path}"

    # Install binaries and libraries to specified directory.
    mv "${unzip_path}/nvim-linux64/bin" "${nvim_install_dir}/bin"
    mv "${unzip_path}/nvim-linux64/lib" "${nvim_install_dir}/lib"
    mv "${unzip_path}/nvim-linux64/share" "${nvim_install_dir}/share"

    echo "The neovim is installed into ${nvim_install_dir}"
    "${nvim_install_dir}"/bin/nvim --version
}

function install_neovim_arm64() {
    bash "${cur_dir}"/build_nvim.sh "${HOME}/.neovim/"
}

function copy_nvim_config() {
    local nvim_config_dir="${emmett_root_path}/config/nvim"
    [[ ! -d ${nvim_config_dir} ]] && echo "Can't find 'nvim' in emmett2020/emmett" && exit 1

    local nvim_config_install_dir="${HOME}/.config/nvim"
    [[ -d "${nvim_config_install_dir}" ]] && mv "${nvim_config_install_dir}" "${nvim_config_install_dir}.backup"
    mkdir -p "${nvim_config_install_dir}"
    cp -r "${nvim_config_dir}/"* "${nvim_config_install_dir}/"
}

##############################################
#               entrypoint
##############################################
arch=$(uname -m)
[[ "${arch}" == "x86_64" ]] && install_neovim_x86_64
[[ "${arch}" == "aarch64" ]] && install_neovim_arm64
copy_nvim_config
