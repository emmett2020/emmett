#!/bin/bash
: << 'COMMENT'
|------------------------------|------------------------------|
|         🎃 item              |        👇 explanation        |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
|          dependencies        |              No              |
|------------------------------|------------------------------|
COMMENT
set -euo pipefail

NONINTERACTIVE=1 CI=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> "${HOME}"/.profile
# shellcheck disable=all
echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> "${HOME}"/.profile
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
brew --version
brew update
