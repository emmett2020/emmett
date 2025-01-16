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
trap "rm -rf ${temp_dir}" EXIT

link_oh_my_zsh=https://install.ohmyz.sh/
link_zsh_syntax_highlighting=https://github.com/zsh-users/zsh-syntax-highlighting
link_zsh_auto_suggestions=https://github.com/zsh-users/zsh-autosuggestions
link_powerlevel10k=https://gitee.com/romkatv/powerlevel10k.git
link_eza="https://github.com/eza-community/eza/releases/download/v0.19.4/eza_x86_64-unknown-linux-gnu.tar.gz"
link_chroma="https://github.com/alecthomas/chroma/releases/download/v2.14.0/chroma-2.14.0-linux-amd64.tar.gz"

cur_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
omz_custom="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

dir_oh_my_zsh="${HOME}/.oh-my-zsh"
dir_zsh_syntax_highlighting="${omz_custom}/plugins/zsh-syntax-highlighting"
dir_zsh_auto_suggesstions="${omz_custom}/plugins/zsh-autosuggestions"
dir_powerlevel10k="${omz_custom}/themes/powerlevel10k"

emmett_path="${cur_dir}/.."
zshrc_path="${emmett_path}/configs/zshrc/daily"

[[ ! -f ${zshrc_path} ]]  && "Can't find 'daily' in emmett2020/emmett" && exit 1
[[ -f "${HOME}/.zshrc" ]] && "Remove ${HOME}/.zshrc first" && exit 1

function install_omz() {
	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  [[ -d "${dir_oh_my_zsh}" ]]               && rm -r "${dir_oh_my_zsh}"
  [[ -d "${dir_powerlevel10k}" ]]           && rm -r "${dir_powerlevel10k}"
  [[ -d "${dir_zsh_auto_suggesstions}" ]]   && rm -r "${dir_zsh_auto_suggesstions}"
  [[ -d "${dir_zsh_syntax_highlighting}" ]] && rm -r "${dir_zsh_syntax_highlighting}"

  source "${cur_dir}/detect_os.sh"
  if [[ "${OS}" == "MacOS" ]]; then
    brew install zsh chroma eva
  elif [[ "${OS}" == "Ubuntu" ]]; then
    sudo apt install -y zsh

    wget "${link_eza}" -O "${tmp}/eva.tar.gz"
    tar -xzf "${tmp}/eva.tar.gz" -C "${tmp}"
    mv "${tmp}/eza" "/usr/local/bin"

    wget "${link_chroma}" -O "${tmp}/chroma.tar.gz"
    tar -xzf "${tmp}/chroma.tar.gz" -C "${tmp}"
    mv "${tmp}/chroma" "/usr/local/bin"
  else
    exit 1
  fi

  # 2. Install oh my zsh
  wget "${link_oh_my_zsh}" -O "${tmp}/oh_my_zsh.sh"
  bash "${tmp}/oh_my_zsh.sh" --unattended

  # 3. Install plugins and themes
  git clone "${link_zsh_syntax_highlighting}" "${dir_zsh_syntax_highlighting}"
  git clone "${link_zsh_auto_suggestions}"    "${dir_zsh_auto_suggesstions}"
  git clone --depth=1 "${link_powerlevel10k}" "${dir_powerlevel10k}"

  # 6. Copy zshrc from emmett repo
  cp "${zshrc_path}" ~/.zshrc

  rm -rf ${tmp}
}

check_zshrc
install_omz

echo "Successfully installed zsh, omz, themes, config and plugins."
echo "Please use: source ~/.zshrc to apply newest config."
