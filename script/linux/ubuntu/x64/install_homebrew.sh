#!/bin/bash
cat << END
Install homebrew use official scripts.
NOTE: This script doesn't support arm machine.
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
END
set -euo pipefail

NONINTERACTIVE=1 CI=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> "${HOME}/.profile"
# shellcheck disable=SC2016,SC2086
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ${HOME}/.profile
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
brew --version
brew update
