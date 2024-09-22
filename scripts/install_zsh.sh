#!/bin/bash
set -e

# 1. /path-to-emmett/configs/zshrc/daily must be existed.

link_oh_my_zsh=https://install.ohmyz.sh/
link_zsh_syntax_highlighting=https://github.com/zsh-users/zsh-syntax-highlighting
link_zsh_auto_suggestions=https://github.com/zsh-users/zsh-autosuggestions
link_powerlevel10k=https://gitee.com/romkatv/powerlevel10k.git
link_eza="https://github.com/eza-community/eza/releases/download/v0.19.4/eza_x86_64-unknown-linux-gnu.tar.gz"

cur_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
omz_custom="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

dir_oh_my_zsh="${HOME}/.oh-my-zsh"
dir_zsh_syntax_highlighting="${omz_custom}/plugins/zsh-syntax-highlighting"
dir_zsh_auto_suggesstions="${omz_custom}/plugins/zsh-autosuggestions"
dir_powerlevel10k="${omz_custom}/themes/powerlevel10k"

emmett_path="${cur_dir}/.."
zshrc_path="${emmett_path}/configs/zshrc/daily"


function check_zshrc() {
  if [[ ! -f "${zshrc_path}" ]]; then
    echo "  Can't find .zshrc file in emmett2020/emmett repo. Path of .zshrc: ${zshrc_path}"
    exit 1
  fi
  if [[ -f "${HOME}/.zshrc" ]]; then
    echo "  Remove or save ${HOME}/.zshrc first."
    exit 1
  fi
}

function print_hint() {
  echo -e "  ......\n\n"
  echo "  Installed zsh, config and plugins(needs sudo permission)"
  echo "  oh-my-zsh download  link: ${link_oh_my_zsh}"
  echo "  oh-my-zsh installed path: ${dir_oh_my_zsh}"
  echo "  zsh-syntax-highlighting download  link: ${link_zsh_syntax_highlighting}"
  echo "  zsh-syntax-highlighting installed path: ${dir_zsh_syntax_highlighting}"
  echo "  zsh-autosuggestions download  link: ${link_zsh_auto_suggestions}"
  echo "  zsh-autosuggestions installed path: ${dir_zsh_auto_suggesstions}"
  echo "  powerlevel10k download  link: ${link_powerlevel10k}"
  echo "  powerlevel10k installed path: ${dir_powerlevel10k}"
  echo "  eza download link: ${link_eza}"
  echo "  Successfully installed zsh, omz, themes, config and plugins."
  echo "  Please use: source ~/.zshrc to apply newest config."
}

function install_zsh() {
	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  [[ -d "${dir_oh_my_zsh}" ]]               && rm -r "${dir_oh_my_zsh}"
  [[ -d "${dir_powerlevel10k}" ]]           && rm -r "${dir_powerlevel10k}"
  [[ -d "${dir_zsh_auto_suggesstions}" ]]   && rm -r "${dir_zsh_auto_suggesstions}"
  [[ -d "${dir_zsh_syntax_highlighting}" ]] && rm -r "${dir_zsh_syntax_highlighting}"

  # 1. Install zsh
  sudo apt install -y zsh

  # 2. Install oh my zsh
  wget "${link_oh_my_zsh}" -O "${tmp}/oh_my_zsh.sh"
  bash "${tmp}/oh_my_zsh.sh" --unattended

  # 3. Install plugins and themes
  git clone "${link_zsh_syntax_highlighting}" "${dir_zsh_syntax_highlighting}"
  git clone "${link_zsh_auto_suggestions}"    "${dir_zsh_auto_suggesstions}"
  git clone --depth=1 "${link_powerlevel10k}" "${dir_powerlevel10k}"

  # 4. Install eva
  wget "${link_eza}" -O "${tmp}/eva.tar.gz"
  tar -xzf "${tmp}/eva.tar.gz" -C "${tmp}"
  ls ${tmp}
  mv "${tmp}/eza" "/usr/local/bin"

  # 5. Copy zshrc from emmett repo
  cp "${zshrc_path}" ~/.zshrc

  rm -rf ${tmp}
}

check_zshrc
install_zsh
print_hint
