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

# We put this check here rather than install_nvim.sh since this check is too
# strict but may not confluence use.
function validate_daily() {
  # Validate nvim
  echo "::group:: validate nvim"
  ${HOME}/.neovim/bin/nvim --version
  ${HOME}/.neovim/bin/nvim --headless -c "TSUpdate query" -c "checkhealth" -c "w!health.log" -c"qa"
  cat health.log
  grep "\- ERROR" health.log | while IFS= read -r line; do
    if echo "$line" | grep -q "command failed: infocmp"; then
      continue
    else
      echo "Health check of neovim failed"
      exit 1
    fi
  done
  rm health.log
  echo "::endgroup::"
}

validate_daily
