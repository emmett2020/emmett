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
	if [ -n "$(uname -a | grep Ubuntu)" ]; then
	  SYSTEM_TYPE="Ubuntu"
	  SYSTEM_VERSION=$(cat /etc/os-release | grep VERSION_ID | awk -F\" '{print $2}')
	  SYSTEM_ARCH=$(uname -m)
	elif [ -n "$(uname -a | grep Darwin)" ]; then
	  SYSTEM_TYPE="Macos"
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
  bash "${DAILY_SCRIPT_DIR}"/install_zsh_plugins.sh
}

# This script only support enhanced getopt version.
function check_getopt_version() {
	getopt -T &>/dev/null
	[ $? -ne 4 ] && {
		echo "Only support enhanced getopt version."
		exit 1
	}
}

# Check that this script works in the current environment.
function check_env() {
	# check_getopt_version
  echo "${SYSTEM_TYPE}"
	get_system_info
	if [[ "$SYSTEM_TYPE" != "Ubuntu" ]]; then
		echo "Only support Ubuntu currently."
		exit 1
	fi
}

function help() {
	echo "------------------------------------------------------------"
	echo "    Enjoy your life                                        "
	echo "------------------------------------------------------------"
	echo "Usage: daily.sh -i command"
	echo "Examples:"
	echo "1. daily.sh -i daily" echo "    Install stable version of daily commands collection."
	echo "Options:"
	echo "-h, --help      Display this usage."
	echo "-i, --install   Install specific command."
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

# Then, get user inputed options.
# $@ is all command line parameters passed to the script.
# -o is for short options like -v
# -l is for long options with double dash like --version
# the comma separates different long options
options=$(getopt -o "+hi:" -l "install:,help" -- "$@")
[[ $? -ne 0 ]] && {
	echo "Try '$0 --help' for more information."
	exit 1
}
eval set -- "$options"

while true; do
	case "$1" in
	-i | --install)
		COMMAND="$2"
		shift 2
		;;
	-h | --help)
		help
		exit 0
		;;
	--)
		shift
		break
		;;
	*)
		help
		exit 1
		;;
	esac
done


# Last, execute command.
if [[ "${COMMAND}" == "daily" ]]; then
  install_daily_commands
else
  option_not_supported_now
fi
