#!/bin/bash

####################################
# @author: emmettzhang2020@outlook.com
# @introduction: Used for daily develop.
# ##################################

set -e

DAILY_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
SYSTEM_TYPE="unknown"
SYSTEM_VERSION=""
SYSTEM_ARCH=""

# Get system informations.
function get_system_info() {
	if grep -q "Ubuntu" "/etc/os-release" ; then
	  SYSTEM_TYPE="Ubuntu"
	  SYSTEM_VERSION=$(cat /etc/os-release | grep VERSION_ID | awk -F\" '{print $2}')
	  SYSTEM_ARCH=$(uname -m)
	fi
}

# Use `which` to check whether there exists specific commands.
function exist_command() {
	local cmd="$1"
	which "$cmd" >/dev/null 2>&1
	if [ $? -eq 0 ]; then
		return 0
	fi
	return 1
}


# Daily commands are a collection of commands that support daily code
# development. Versions of commands should be stable.
function install_daily_commands() {
  bash "${DAILY_SCRIPT_DIR}"/install_cmake.sh
  bash "${DAILY_SCRIPT_DIR}"/install_codelldb.sh
  bash "${DAILY_SCRIPT_DIR}"/install_cppdbg.sh
  bash "${DAILY_SCRIPT_DIR}"/install_fdfind.sh
  bash "${DAILY_SCRIPT_DIR}"/install_lazygit.sh
  bash "${DAILY_SCRIPT_DIR}"/install_nvim.sh
  bash "${DAILY_SCRIPT_DIR}"/install_ripgrep.sh
  bash "${DAILY_SCRIPT_DIR}"/install_zsh.sh
}

# Check that this script works in the current environment.
function check_env() {
  echo "${SYSTEM_TYPE}"
	get_system_info
	if [[ "$SYSTEM_TYPE" != "Ubuntu" ]]; then
		echo "Only support Ubuntu currently."
		exit 1
	fi
}

function option_not_supported_now() {
	echo "This option is not not supported now."
	exit 1
}

############################################################
#####                     Start                        #####
############################################################

# Firstly, check local envrionment.
check_env

# Then, execute command.
install_daily_commands
