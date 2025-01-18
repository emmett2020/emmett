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

