#!/bin/bash
: << 'COMMENT'
|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |              Yes              |
|------------------------------|-------------------------------|
|          dependencies        |           ${emmett}           |
|------------------------------|-------------------------------|
|          Archecture          |         x86-64 / arm64        |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

CUR_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

bash "${CUR_SCRIPT_DIR}"/install_cmake.sh
bash "${CUR_SCRIPT_DIR}"/install_fdfind.sh
bash "${CUR_SCRIPT_DIR}"/install_lazygit.sh
bash "${CUR_SCRIPT_DIR}"/install_ripgrep.sh
bash "${CUR_SCRIPT_DIR}"/install_nvim.sh
bash "${CUR_SCRIPT_DIR}"/install_zsh.sh

function validate_daily() {
  POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true zsh

  cmake --version
  fd --version
  lazygit --version
  rg --version

  # Validate nvim
  echo "::group:: validate nvim"
  nvim --version
  nvim --headless -c "checkhealth" -c "w\!health.log" -c"qa"
  cat health.log
  if grep -q "- ERROR" health.log; then
    exit 1
  fi
  echo "::endgroup::"

  # Validate zsh
}

validate_daily
