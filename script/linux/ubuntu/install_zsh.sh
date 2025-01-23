#!/bin/bash
: << 'COMMENT'
|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |              Yes              |
|------------------------------|-------------------------------|
|          dependencies        |  ${emmett}/config/zshrc/daily |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

temp_dir=$(mktemp -d)
trap 'rm -rf ${temp_dir}' EXIT

function copy_zshrc() {
    cur_dir=$(
        cd "$(dirname "${BASH_SOURCE[0]}")"
        pwd
    )
    local emmett_root_path="${cur_dir}/../../.."
    local zshrc_path="${emmett_root_path}/config/zshrc/daily"
    [[ ! -f ${zshrc_path} ]] && echo "Can't find 'daily' in emmett2020/emmett" && exit 1
    [[ -f "${HOME}/.zshrc" ]] && cp "${HOME}/.zshrc" "${HOME}/.zshrc.backup"
    cp "${zshrc_path}" "${HOME}/.zshrc"
}

function install_zsh() {
    sudo apt install -y zsh
}

function install_oh_my_zsh() {
    [[ -d "${HOME}/.oh-my-zsh" ]] && mv "${HOME}"/.oh-my-zsh "${HOME}"/.oh-my-zsh.backup
    wget "https://install.ohmyz.sh/" -O "${temp_dir}/oh_my_zsh.sh"
    bash "${temp_dir}/oh_my_zsh.sh" --unattended
}

omz_custom_dir="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

function install_zsh_syntax_highlighting() {
    local install_dir="${omz_custom_dir}/plugins/zsh-syntax-highlighting"
    [[ -d "${install_dir}" ]] && rm -rf "${install_dir}"
    git clone https://github.com/zsh-users/zsh-syntax-highlighting "${install_dir}"
}

function install_zsh_auto_suggestions() {
    local install_dir="${omz_custom_dir}/plugins/zsh-autosuggestions"
    [[ -d "${install_dir}" ]] && rm -r "${install_dir}"
    git clone https://github.com/zsh-users/zsh-autosuggestions "${install_dir}"
}

function install_powerlevel_10k() {
    local install_dir="${omz_custom_dir}/themes/powerlevel10k"
    [[ -d "${install_dir}" ]] && rm -r "${install_dir}"
    git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git "${install_dir}"
}

function install_eza() {
    arch=$(uname -m)
    local version=0.20.17
    local link="https://github.com/eza-community/eza/releases/download/v${version}/eza_${arch}-unknown-linux-gnu.tar.gz"

    wget "${link}" -O "${temp_dir}/eva.tar.gz"
    tar -xzf "${temp_dir}/eva.tar.gz" -C "${temp_dir}"
    sudo mv "${temp_dir}/eza" "/usr/local/bin"
}

function install_chroma() {
    arch=$(uname -m)
    [[ "${arch}" == "aarch64" ]] && arch="arm64"
    [[ "${arch}" == "x86_64" ]] && arch="amd64"

    local version=2.15.0
    local link="https://github.com/alecthomas/chroma/releases/download/v${version}/chroma-${version}-linux-${arch}.tar.gz"

    wget "${link}" -O "${temp_dir}/chroma.tar.gz"
    tar -xzf "${temp_dir}/chroma.tar.gz" -C "${temp_dir}"
    sudo mv "${temp_dir}/chroma" "/usr/local/bin"
}

##############################################
#               entrypoint
##############################################
install_zsh
install_oh_my_zsh
install_zsh_syntax_highlighting
install_zsh_auto_suggestions
install_powerlevel_10k
install_eza
install_chroma
copy_zshrc

echo "Successfully installed zsh, omz, config, themes and plugins."
echo "Please use: source ~/.zshrc to apply newest config."
