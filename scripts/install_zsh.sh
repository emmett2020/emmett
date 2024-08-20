#!/bin/bash
set -e

# 1. Always install zsh default version.
# 2. /path-to-emmett/configs/zshrc/daily must be existed.

CUR_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
OH_MY_ZSH_LINK=https://install.ohmyz.sh/
ZSH_SYNTAX_HIGHLIGHTING_LINK=https://github.com/zsh-users/zsh-syntax-highlighting
ZSH_SYNTAX_HIGHLIGHTING_INSTALL_PATH="${HOME}/.zsh/zsh-syntax-highlighting"
ZSH_AUTOSUGGESTIONS_LINK=https://github.com/zsh-users/zsh-autosuggestions
ZSH_AUTOSUGGESTIONS_INSTALL_PATH="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}/plugins/zsh-autosuggestions"
POWERLEVEL10k_LINK=https://gitee.com/romkatv/powerlevel10k.git
POWERLEVEL10k_INSTALL_PATH="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k"

function install_zsh() {
  echo "  Installing zsh (needs sudo permission)"
  echo "  oh-my-zsh: ${OH_MY_ZSH_LINK}"
  echo "  zsh-syntax-highlighting link: ${ZSH_SYNTAX_HIGHLIGHTING_LINK}"
  echo "  zsh-syntax-highlighting will be installed into: ${ZSH_SYNTAX_HIGHLIGHTING_INSTALL_PATH}"
  echo "  zsh-autosuggestions link: ${ZSH_AUTOSUGGESTIONS_LINK}"
  echo "  zsh-autosuggestions will be installed into: ${ZSH_AUTOSUGGESTIONS_INSTALL_PATH}"
  echo "  powerlevel10k link: ${POWERLEVEL10k_LINK}"
  echo "  powerlevel10k will be installed into: ${POWERLEVEL10k_INSTALL_PATH}"
  echo "  Other plugins used and supported by zsh: git z extract"
  echo -e "  ......\n\n"

  sudo apt install zsh
  chsh -s /bin/zsh
  sh -c "$(wget -O- ${OH_MY_ZSH_LINK})"
  git clone "${ZSH_SYNTAX_HIGHLIGHTING_LINK}" "${ZSH_SYNTAX_HIGHLIGHTING_INSTALL_PATH}"
  git clone "${ZSH_AUTOSUGGESTIONS_LINK}"     "${ZSH_AUTOSUGGESTIONS_INSTALL_PATH}"
  git clone --depth=1 "${POWERLEVEL10k_LINK}" "${POWERLEVEL10k_INSTALL_PATH}"

  local emmett_path="${CUR_SCRIPT_DIR}/.."
  local zshrc_path="${emmett_path}/configs/zshrc/daily"

  if [[ ! -f "${zshrc_path}" ]]; then
    echo "  Cann't find .zshrc file in emmett repo."
    echo "  .zshrc path: ${zshrc_path} "
    exit 1
  fi

  rm -f ~/.zshrc
  cp "${zshrc_path}" ~/.zshrc
  /bin/zsh -c "source ~/.zshrc"
}

install_zsh
