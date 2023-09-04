#!/bin/bash

####################################
# @author: xiaomingZhang2020@outlook.com
# @introduction: Used for daily develop.
# ##################################

# Use -V to open debug mode.
VERBOSE=0

# Supported command and it's version.
RIPGREP_STABLE_VERSION="13.0.0"
FDFIND_STABLE_VERSION="8.7.0"
LAZYGIT_STABLE_VERSION="0.40.2"
CPPDBG_STABLE_VERSION="1.17.4"
CODELLDB_STABLE_VERSION="1.9.2"
NEOVIM_STABLE_VERSION="0.9.1"

# Get system informations.
function get_system_infos() {
	SYSTEM_TYPE="unknown"
	SYSTEM_VERSION=""
	SYSTEM_ARCH=""
	if [ -n "$(uname -a | grep Ubuntu)" ]; then
		SYSTEM_TYPE="Ubuntu"
		SYSTEM_VERSION=$(cat /etc/os-release | grep VERSION_ID | awk -F\" '{print $2}')
		SYSTEM_ARCH=$(uname -m)
	elif [ -n "$(uname -a | grep Darwin)" ]; then
		FONT_VERSION="$2"
		SYSTEM_TYPE="Macos"
	fi
}

# Use `which` to check whether there exists specific commands.
function exist_command() {
	local cmd="$1"
	which "$cmd" >/dev/null 2>&1
	if [ $? -eq 0 ]; then
		# In shell programming, 0 is success.
		return 0
	fi
	return 1
}

##################################################################
#                                                        [ripgrep]

# Get ripgrep version installed at this machine.
# Must fill INSTALLED_VERSION since it's used by outside.
function get_ripgrep_version() {
	INSTALLED_VERSION=$(rg --version | grep ripgrep | awk '{print $2}')
	return $?
}

function exist_ripgrep() {
	exist_command rg
	return $?
}

function install_ripgrep() {
	local version=$1
	local reinstall=$2

	if exist_ripgrep; then
		INSTALLED_VERSION=""
		get_ripgrep_version
		if [ "$INSTALLED_VERSION" == "$version" ]; then
			echo " The ripgrep $INSTALLED_VERSION already exists. No need to install again."
			rg --version
			return 0
		fi
		if [ "$reinstall" == "0" ]; then
			echo "The ripgrep $INSTALLED_VERSION already exists. Try to reinstall it?"
			return 1
		fi
	fi

	local link="https://github.com/BurntSushi/ripgrep/releases/download/${version}/ripgrep_${version}_amd64.deb"
	local tmp="/tmp/ripgrep.deb"
	if [ "$VERBOSE" == "0" ]; then
		echo " Installing ripgrep $version ......"
		echo " Link: $link"
		{
			wget $link -O $tmp
			sudo dpkg -i $tmp
			rm $tmp
		} &>/dev/null
	else
		wget $link -O $tmp
		sudo dpkg -i $tmp
		rm $tmp
	fi
	rg --version
}

function uninstall_ripgrep() {
	if exist_command rg; then
		sudo dpkg -r ripgrep
		return $?
	fi
	echo " The rg doesn't exist."
	return 1
}

##################################################################
#                                                         [fdfind]

function get_fdfind_version() {
	INSTALLED_VERSION=$(fd --version | grep fd | awk '{print $2}')
	return $?
}

function exist_fdfind() {
	exist_command fd
	return $?
}

function install_fdfind() {
	local version=$1
	local reinstall=$2

	if exist_fdfind; then
		INSTALLED_VERSION=""
		get_fdfind_version
		if [ "$INSTALLED_VERSION" == "$version" ]; then
			echo " The fd $INSTALLED_VERSION already exists. No need to install again."
			fd --version
			return 0
		fi
		if [ "$reinstall" == "0" ]; then
			echo "The fd $INSTALLED_VERSION already exists. Try to reinstall it?"
			return 1
		fi
	fi

	local link="https://github.com/sharkdp/fd/releases/download/v${version}/fd_${version}_amd64.deb"
	local tmp="/tmp/fdfind.deb"
	if [ "$VERBOSE" == "0" ]; then
		echo " Installing fdfind $version ......"
		echo " Link: $link"
		{
			wget $link -O $tmp
			sudo dpkg -i $tmp
			rm $tmp
		} &>/dev/null
	else
		wget $link -O $tmp
		sudo dpkg -i $tmp
		rm $tmp
	fi
	fd --version
}

function uninstall_fdfind() {
	if exist_fdfind; then
		sudo dpkg -r fd
		return $?
	fi
	echo " The fd doesn't exist."
	return 1
}

##################################################################
#                                                        [lazygit]
function get_lazygit_version() {
	INSTALLED_VERSION=$(lazygit --version | awk -F, '{print $4}' | awk -F= '{print $2}')
	return $?
}

function exist_lazygit() {
	exist_command lazygit
	return $?
}

function install_lazygit() {
	local version=$1
	local reinstall=$2

	if exist_lazygit; then
		INSTALLED_VERSION=""
		get_lazygit_version
		if [ "$INSTALLED_VERSION" == "$version" ]; then
			echo " The lazygit $INSTALLED_VERSION already exists. No need to install again."
			lazygit --version
			return 0
		fi
		if [ "$reinstall" == "0" ]; then
			echo "The lazygit $INSTALLED_VERSION already exists. Try to reinstall it?"
			return 1
		fi
	fi

	local link="https://github.com/jesseduffield/lazygit/releases/download/v${version}/lazygit_${version}_Linux_x86_64.tar.gz"
	local tmp_gz="/tmp/lazygit.tar.gz"
	local tmp="/tmp/lazygit"
	local install_path="/usr/local/bin"

	if [ "$VERBOSE" == "0" ]; then
		echo " Installing lazygit $version ......"
		echo " Link: $link"
		{
			wget $link -O $tmp_gz
			sudo mkdir $tmp
			sudo tar xf $tmp_gz -C $tmp
			sudo install "$tmp/lazygit" $install_path
			sudo rm -rf $tmp
			sudo rm $tmp_gz
		} &>/dev/null
	else
		wget $link -O $tmp_gz
		sudo mkdir $tmp
		sudo tar xf $tmp_gz -C $tmp
		sudo install "$tmp/lazygit" $install_path
		sudo rm -rf $tmp
		sudo rm $tmp_gz
	fi
	lazygit --version
}

function uninstall_lazygit() {
	if exist_command lazygit; then
		sudo rm /usr/local/bin/lazygit
		return $?
	fi
	echo " The lazygit doesn't exist."
	return 1
}

##################################################################
#                                                         [cppdbg]
CPPDBG_INSTALL_PATH="$HOME/.local/share/nvim/cppdbg/"

function exist_cppdbg() {
	# Just use path of OpenDebugAD7 to check wheter cppdbg exists.
	local p="$CPPDBG_INSTALL_PATH/extension/debugAdapters/bin/OpenDebugAD7"
	if [[ -e $p ]]; then
		return 0
	else
		return 1
	fi
}

# We don't have a good way to get cppdbg version.
function get_cppdbg_version() {
	INSTALLED_VERSION="---"
}

function uninstall_cppdbg() {
	if exist_cppdbg; then
		sudo rm -r $CPPDBG_INSTALL_PATH
		return $?
	fi
	echo " The cppdbg doesn't exist."
	return 1
}

function install_cppdbg() {
	local version=$1
	local reinstall=$2

	echo "WARN: There is no good way to get the cppdbg version, so always install/reinstall it."
	uninstall_cppdbg &>/dev/null

	local link="https://github.com/microsoft/vscode-cpptools/releases/download/v${version}/cpptools-linux.vsix"
	local vsix_path="/tmp/cpptools-linux.vsix"
	local unzip_path="/tmp/cppdbg/"

	if [ "$VERBOSE" == "0" ]; then
		echo " Installing cppdbg $version ......"
		echo " Link: $link"
		{
			wget $link -O $vsix_path
			unzip $vsix_path -d $unzip_path
			mkdir -p $CPPDBG_INSTALL_PATH
			mv "$unzip_path/extension" "$CPPDBG_INSTALL_PATH"
			chmod +x "$CPPDBG_INSTALL_PATH/extension/debugAdapters/bin/OpenDebugAD7"
			sudo rm $vsix_path
			sudo rm -r $unzip_path
		} &>/dev/null
	else
		wget $link -O $vsix_path
		unzip $vsix_path -d $unzip_path
		mkdir -p $CPPDBG_INSTALL_PATH
		mv "$unzip_path/extension" "$CPPDBG_INSTALL_PATH"
		chmod +x "$CPPDBG_INSTALL_PATH/extension/debugAdapters/bin/OpenDebugAD7"
		sudo rm $vsix_path
		sudo rm -r $unzip_path
	fi
	echo "Installed cppdbg $version."
}
##################################################################
#                                                       [codelldb]
CODELLDB_INSTALL_PATH="$HOME/.local/share/nvim/codelldb/"

# We don't have method to get cppdbg version.
function get_codelldb_version() {
	INSTALLED_VERSION="---"
}

function exist_codelldb() {
	local p="$CODELLDB_INSTALL_PATH/extension/adapter/codelldb"
	if [[ -e $p ]]; then
		return 0
	else
		return 1
	fi
}

function uninstall_codelldb() {
	if exist_codelldb; then
		sudo rm -r $CODELLDB_INSTALL_PATH
		return $?
	fi
	echo " The codelldb doesn't exist."
	return 1
}

function install_codelldb() {
	local version=$1
	local reinstall=$2

	echo "WARN: There is no good way to get the codelldb version, so always install/reinstall it."
	uninstall_codelldb &>/dev/null

	local link="https://github.com/vadimcn/codelldb/releases/download/v${version}/codelldb-x86_64-linux.vsix"
	local vsix_path="/tmp/codelldb.vsix"
	local unzip_path="/tmp/codelldb/"

	if [ "$VERBOSE" == "0" ]; then
		echo " Installing codelldb $version ......"
		echo " Link: $link"
		{
			wget $link -O $vsix_path
			unzip $vsix_path -d $unzip_path
			mkdir -p $CODELLDB_INSTALL_PATH
			mv "$unzip_path/extension" $CODELLDB_INSTALL_PATH
			chmod +x "$CODELLDB_INSTALL_PATH/extension/adapter/codelldb"
			sudo rm $vsix_path
			sudo rm -r $unzip_path
		} &>/dev/null
	else
		wget $link -O $vsix_path
		unzip $vsix_path -d $unzip_path
		mkdir -p $CODELLDB_INSTALL_PATH
		mv "$unzip_path/extension" $CODELLDB_INSTALL_PATH
		chmod +x "$CODELLDB_INSTALL_PATH/extension/adapter/codelldb"
		sudo rm $vsix_path
		sudo rm -r $unzip_path
	fi
	echo "Installed codelldb $version."
}

##################################################################
#                                                         [neovim]

NEOVIM_INSTALL_PATH="$HOME/.neovim/"

function get_neovim_version() {
	INSTALLED_VERSION=$(nvim --version | grep NVIM | awk -Fv '{print $2}')
	return $?
}

function exist_neovim() {
	exist_command nvim
	return $?
}

function install_neovim() {
	local version=$1
	local reinstall=$2
	if exist_neovim; then
		INSTALLED_VERSION=""
		get_neovim_version
		if [ "$INSTALLED_VERSION" == "$version" ]; then
			echo " The neovim $INSTALLED_VERSION already exists. No need to install again."
			nvim --version
			return 0
		fi
		if [ "$reinstall" == "0" ]; then
			echo "The neovim $INSTALLED_VERSION already exists. Try to reinstall it?"
			return 1
		fi
	fi

	link="https://github.com/neovim/neovim/releases/download/v${version}/nvim-linux64.tar.gz"
	local tmp_path="/tmp/nvim-linux.tar.gz"
	local unzip_path="/tmp/neovim"
	if [ -d "$NEOVIM_INSTALL_PATH" ]; then
		sudo rm -rf "$NEOVIM_INSTALL_PATH"
	fi

	mkdir -p "$NEOVIM_INSTALL_PATH"

	if [ "$VERBOSE" == "0" ]; then
		echo " Installing neovim $version ......"
		echo " Link: $link"
		{
			wget $link -O $tmp_path
			if [ ! -d "$unzip_path" ]; then
				mkdir $unzip_path
			fi
			tar -xzf $tmp_path -C $unzip_path
			mv "$unzip_path/nvim-linux64/bin" "$NEOVIM_INSTALL_PATH/bin"
			mv "$unzip_path/nvim-linux64/lib" "$NEOVIM_INSTALL_PATH/lib"
			mv "$unzip_path/nvim-linux64/share" "$NEOVIM_INSTALL_PATH/share"
			sudo rm $tmp_path
			sudo rm -r $unzip_path
		} &>/dev/null
	else
		wget $link -O $tmp_path
		if [ ! -d "$unzip_path" ]; then
			mkdir $unzip_path
		fi
		tar -xzf $tmp_path -C $unzip_path
		mv "$unzip_path/nvim-linux64/bin" "$NEOVIM_INSTALL_PATH/bin"
		mv "$unzip_path/nvim-linux64/lib" "$NEOVIM_INSTALL_PATH/lib"
		mv "$unzip_path/nvim-linux64/share" "$NEOVIM_INSTALL_PATH/share"
		sudo rm $tmp_path
		sudo rm -r $unzip_path
	fi
	echo ' You should add '$HOME/.neovim/bin' to your $PATH and enable it use follow steps:'
	echo 'export PATH="$HOME/.neovim/bin:$PATH'
	echo 'source "$HOME/.zshrc"'
	return 0
}

function uninstall_neovim() {
	if exist_neovim; then
		sudo rm -rf "$NEOVIM_INSTALL_PATH"
		return $?
	fi
	echo " The nvim doesn't exist."
	return 1
}

##################################################################
#                                                          [daily]

# Daily commands are a collection of commands that
# support our daily code development. Because daily
# commands are a collection, it is not possible to
# provide a separate version installation for a
# single command. Version can only be stable, and
# this applies to all internal commands.
function install_daily_commands() {
	local reinstall=$2
	install_ripgrep $RIPGREP_STABLE_VERSION $reinstall
	install_fdfind $FDFIND_STABLE_VERSION $reinstall
	install_lazygit $LAZYGIT_STABLE_VERSION $reinstall
	install_cppdbg $CPPDBG_STABLE_VERSION $reinstall
	install_codelldb $CODELLDB_STABLE_VERSION $reinstall
}

function uninstall_daily_commands() {
	uninstall_ripgrep
	uninstall_fdfind
	uninstall_lazygit
	uninstall_cppdbg
	uninstall_codelldb
}

function exist_daily() {
	exist_ripgrep &&
		exist_fdfind &&
		exist_lazygit &&
		exist_cppdbg &&
		exist_codelldb && {
		return 0
	}
	return 1
}

function get_daily_version() {
	INSTALLED_VERSION="---"
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
function check() {
	check_getopt_version
	get_system_infos
	if [[ "$SYSTEM_TYPE" != "Ubuntu" ]]; then
		echo "Only support Ubuntu currently."
		exit 1
	fi
}

############################################################
# Declare needed commands associative array.
declare -A ripgrep_info=(
	["stable_version"]="$RIPGREP_STABLE_VERSION"
	["func_install"]=install_ripgrep
	["func_uninstall"]=uninstall_ripgrep
	["func_get_installed_version"]=get_ripgrep_version
	["func_exist"]=exist_ripgrep
)

declare -A fdfind_info=(
	["stable_version"]="$FDFIND_STABLE_VERSION"
	["func_install"]=install_fdfind
	["func_uninstall"]=uninstall_fdfind
	["func_get_installed_version"]=get_fdfind_version
	["func_exist"]=exist_fdfind
)

declare -A lazygit_info=(
	["stable_version"]="$LAZYGIT_STABLE_VERSION"
	["func_install"]=install_lazygit
	["func_uninstall"]=uninstall_lazygit
	["func_get_installed_version"]=get_lazygit_version
	["func_exist"]=exist_lazygit
)

declare -A cppdbg_info=(
	["stable_version"]="$CPPDBG_STABLE_VERSION"
	["func_install"]=install_cppdbg
	["func_uninstall"]=uninstall_cppdbg
	["func_get_installed_version"]=get_cppdbg_version
	["func_exist"]=exist_cppdbg
)

declare -A codelldb_info=(
	["stable_version"]="$CODELLDB_STABLE_VERSION"
	["func_install"]=install_codelldb
	["func_uninstall"]=uninstall_codelldb
	["func_get_installed_version"]=get_codelldb_version
	["func_exist"]=exist_codelldb
)

declare -A daily_info=(
	["stable_version"]=""
	["func_install"]=install_daily_commands
	["func_uninstall"]=uninstall_daily_commands
	["func_get_installed_version"]=get_daily_version
	["func_exist"]=exist_daily
)

declare -A nvim_info=(
	["stable_version"]="$NEOVIM_STABLE_VERSION"
	["func_install"]=install_neovim
	["func_uninstall"]=uninstall_neovim
	["func_get_installed_version"]=get_neovim_version
	["func_exist"]=exist_neovim
)

declare -A all_commands_info=(
	["rg"]="ripgrep_info"
	["fd"]="fdfind_info"
	["lazygit"]="lazygit_info"
	["cppdbg"]="cppdbg_info"
	["codelldb"]="codelldb_info"
	["daily"]="daily_info"
	["neovim"]="nvim_info"
)

###################################################################
#######################  Fonts    #################################
###################################################################

function use_font_nerd() {
	local subclass="$1"
	local version="$2"
	local link="https://github.com/ryanoasis/nerd-fonts/releases/download/v${version}/${subclass}.zip"
	local tmp_path="/tmp/${subclass}.zip"
	if [ "$VERBOSE" == "0" ]; then
		echo " Installing nerd-fonts ${subclass} ${version} ......"
		echo " Link: $link"
		{
			wget $link -O $tmp_path
			if [ ! -d "$HOME/.fonts/" ]; then
				mkdir -p "$HOME/.fonts/"
			fi
			unzip ${tmp_path} -d "$HOME/.fonts"
			fc-cache -fv
			sudo rm $tmp_path
		} &>/dev/null
	else
		wget $link -O $tmp_path
		if [ ! -d "$HOME/.fonts/" ]; then
			mkdir -p "$HOME/.fonts/"
		fi
		unzip ${tmp_path} -d "$HOME/.fonts"
		fc-cache -fv
		sudo rm $tmp_path
	fi
	echo "Installed nerd-fonts ${subclass} $version."
}

###############################################################################

function help() {
	echo "------------------------------------------------------------"
	echo "    Enjoy your life                                        "
	echo "------------------------------------------------------------"
	echo "Usage: daily.sh -i|r|u command [-v version] [-V]"
	echo "Examples:"
	echo "1. daily.sh -i daily"
	echo "    Install stable version of daily commands collection."
	echo "2. daily.sh -i gdb -v 0.0.1 -V"
	echo "    Install gdb 0.0.1 while show the detailed process."
	echo "4. daily.sh -r cmake -v 3.26.1"
	echo "    Reinstall 3.26.1 version of cmake."
	echo "5. daily.sh -u gdb"
	echo "    Uninstall gdb."
	echo "6. daily.sh --list"
	echo "    Show detailed informations of this environment."
	echo "7. daily.sh -f Haskle -v 3.0.2"
	echo "    Install nerd fonts names Haskle."
	echo ""
	echo "Options:"
	echo "-h, --help      Display this usage."
	echo "-i, --install   Install specific command."
	echo "-r, --reinstall Reinstall specific command."
	echo "-u, --uninstall Uninstall specific command."
	echo "-v, --version   Specific version to install or reinstall."
	echo "-V, --verbose   Run script in verbose mode which will print"
	echo "                out each step of execution."
	echo "-f, --use-font  Use specific font. Will install it firstly."
	echo "-s, --use-shell Use specific shell. Will install it firstly"
	echo "                if not exists. Note that awesome plugins "
	echo "                of this shell will also be installed."
	echo "--list          Show detailed command list informations."
}

function option_not_supported_now() {
	echo "This option is not not supported now."
	exit 1
}

############################################################
#####                     Start                        #####
############################################################

# Check firstly.
check

# $@ is all command line parameters passed to the script.
# -o is for short options like -v
# -l is for long options with double dash like --version
# the comma separates different long options
options=$(
	getopt -o "+hi:r:u:v:lVf:s:" -l "help,\
                     install:,reinstall:,\
                     uninstall:,version:,\
                     verbose,list,\
                     use-font:,use-shell" \
		-- "$@"
)
[ $? -ne 0 ] && {
	echo "Try '$0 --help' for more information."
	exit 1
}
eval set -- "$options"

while true; do
	case "$1" in
	-i | --install)
		MODE="install"
		COMMAND="$2"
		shift 2
		;;
	-r | --reintall)
		MODE="reinstall"
		COMMAND="$2"
		shift 2
		;;
	-u | --uninstall)
		MODE="uninstall"
		COMMAND="$2"
		shift 2
		;;
	-v | --version)
		VERSION="$2"
		shift 2
		;;
	-h | --help)
		help
		exit 0
		;;
	-V | --verbose)
		VERBOSE=1
		shift
		;;
	-l | --list)
		MODE="list"
		shift
		break
		;;
	-f | --use-font)
		MODE="use-font"
		FONT_SUBCLASS="$2"
		shift 2
		echo $FONT_SUBCLASS
		;;
	-s | --use-shell)
		option_not_supported_now
		exit 1
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

if [ "$MODE" == "install" -o "$MODE" == "reinstall" ]; then
	if [ "$MODE" == "reinstall" ]; then
		reinstall_flag=1
	else
		reinstall_flag=0
	fi

	for cmd_name in ${!all_commands_info[*]}; do
		if [ "$cmd_name" == "$COMMAND" ]; then
			declare -n cmd_info="${all_commands_info["$cmd_name"]}"
			stable_version=${cmd_info["stable_version"]}
			install=${cmd_info["func_install"]}
			get_installed_version=${cmd_info["func_get_installed_version"]}

			if [ "$VERSION" == "" ]; then
				VERSION=$stable_version
			fi

			$install $VERSION $reinstall_flag
			exit $?
		fi
	done
	option_not_supported_now
elif [ "$MODE" == "uninstall" ]; then
	for cmd_name in ${!all_commands_info[*]}; do
		if [ "$cmd_name" == "$COMMAND" ]; then
			declare -n cmd_info="${all_commands_info["$cmd_name"]}"
			uninstall=${cmd_info["func_uninstall"]}
			$uninstall
			exit $?
		fi
	done
	echo " Command: $COMMAND doesn't exist or not managed by daily.sh."
	exit 1
elif [ "$MODE" == "list" ]; then
	get_system_infos
	echo "$SYSTEM_TYPE $SYSTEM_VERSION $SYSTEM_ARCH"
	echo "----------------------"
	for cmd_name in ${!all_commands_info[*]}; do
		declare -n cmd_info="${all_commands_info["$cmd_name"]}"
		command_exist=${cmd_info["func_exist"]}
		if $command_exist; then
			INSTALLED_VERSION=""
			get_installed_version=${cmd_info["func_get_installed_version"]}
			$get_installed_version
			printf " %-10s %-10s\n" $cmd_name $INSTALLED_VERSION
		else
			printf " %-10s\n" $cmd_name
		fi
	done
	echo "----------------------"
	echo "(Some items' version aren't shown)"
elif [ "$MODE" == "use-font" ]; then
	if [ "$VERSION" == "" ]; then
		echo "Please provide a specific version of nerd fonts."
		exit 1
	fi
	echo "WARN: Only support nerd-fonts which is enough for daily use."
	echo "NOTE: You can remove $HOME/.fonts directory to remove all fonts."
	use_font_nerd $FONT_SUBCLASS $VERSION
	exit $?
else
	option_not_supported_now
fi
