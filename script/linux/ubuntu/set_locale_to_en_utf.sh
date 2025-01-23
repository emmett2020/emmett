#!/bin/bash
: << 'COMMENT'
Script to update system locale to en_US.UTF-8
|------------------------------|-------------------------------|
|         🎃 item              |        👇 explanation         |
|------------------------------|-------------------------------|
|    needs root permission?    |              Yes              |
|------------------------------|-------------------------------|
COMMENT
set -euo pipefail

sudo apt install language-pack-en-base

echo "Set this variables to your ~/.bashrc or ~/.zshrc and reload shell"
echo "export LANG=en_US.UTF-8"
echo "export LC_ALL=en_US.UTF-8"
echo "export LANGUAGE=en_US.UTF-8"
echo "Then use 'locale' to valid setting is enabled"
