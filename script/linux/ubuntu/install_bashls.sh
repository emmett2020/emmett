#!/bin/bash
: << 'COMMENT'
https://github.com/bash-lsp/bash-language-server
|------------------------------|------------------------------|
|         🎃 item              |        👇 explanation        |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
|          dependencies        |              No              |
|------------------------------|------------------------------|
COMMENT
set -euo pipefail

function command_exists() {
  which "$1" > /dev/null 2>&1
}

if command_exists "npm"; then
  sudo npm i -g bash-language-server
elif command_exists "brew"; then
  brew install bash-language-server
else
  echo "Install brew or npm first to install bash-language-server"
  exit 1
fi

sudo apt install shfmt
sudo apt install shellcheck

bash-language-server --version
shfmt --version
shellcheck --version
