#!/bin/bash
: << 'COMMENT'
Check some basic uses here to avoid obvious fatal error.
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

# Validate CMake
cmake --version

# Validate fdfind
fd --version

# Validate lazygit
lazygit --version

# Validate ripgrep
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
