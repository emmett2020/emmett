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
    which "$1" >/dev/null 2>&1
}

if command_exists "brew"; then
  brew install bash-language-server
  brew install shfmt
  brew install shellcheck
elif command_exists "npm"; then
  npm i -g bash-language-server
  echo "DOING"
  exit 1
else
  echo "Install brew or npm first to install bash-language-server"
  exit 1
fi

bash-language-server --version
shfmt --version
shellcheck --version
