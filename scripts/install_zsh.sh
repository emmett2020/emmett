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

  sudo apt install zsh
  chsh -s /bin/zsh
  [[ -d "${OH_MY_ZSH_INSTALL_PATH}" ]] &&  rm -r "${OH_MY_ZSH_INSTALL_PATH}"
  [[ -f "${HOME}/.zshrc" ]] &&  rm -r "${HOME}/.zshrc"
  sh -c "$(wget -O- ${OH_MY_ZSH_LINK})" --skip-chsh
}

install_zsh
