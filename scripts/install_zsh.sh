#!/bin/bash
set -e

# 1. Always install zsh default version.

CUR_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
OH_MY_ZSH_LINK=https://install.ohmyz.sh/
OH_MY_ZSH_INSTALL_PATH="${HOME}/.oh-my-zsh"

function install_zsh() {
  echo "  Installing zsh (needs sudo permission)"
  echo "  oh-my-zsh link: ${OH_MY_ZSH_LINK}"
  echo "  oh-my-zsh will be installed into: ${OH_MY_ZSH_INSTALL_PATH}"
  echo -e "  ......\n\n"


	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  sudo apt install zsh -y
  chsh -s /bin/zsh
  [[ -d "${OH_MY_ZSH_INSTALL_PATH}" ]] &&  rm -r "${OH_MY_ZSH_INSTALL_PATH}"
  [[ -f "${HOME}/.zshrc" ]] &&  rm -r "${HOME}/.zshrc"

  wget "${OH_MY_ZSH_LINK}" -O "${tmp}/oh_my_zsh.sh"
  bash "${tmp}/oh_my_zsh.sh" --unattended
  rm -rf ${tmp}

  echo "  successfully installed zsh and oh-my-zsh"
}

install_zsh
