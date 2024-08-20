NEOVIM_VERSION="0.10.1"
NEOVIM_INSTALL_PATH="$HOME/.neovim/"
NEOVIM_LINK="https://github.com/neovim/neovim/releases/download/v${NEOVIM_VERSION}/nvim-linux64.tar.gz"

function get_neovim_version() {
	INSTALLED_VERSION=$(nvim --version | grep NVIM | awk -Fv '{print $2}')
	return $?
}


function install_neovim() {
	local tmp_path="/tmp/nvim-linux.tar.gz"
	local unzip_path="/tmp/neovim"

  [[ -d "${NEOVIM_INSTALL_PATH}" ]] && sudo rm -rf ${NEOVIM_INSTALL_PATH}
	mkdir -p "$NEOVIM_INSTALL_PATH"

  echo " Installing neovim $version ......"
  echo " Link: ${NEOVIM_LINK}"
  wget ${NEOVIM_LINK} -O ${tmp_path}
  [[ ! -d "${unzip_path}" ]] && mkdir ${unzip_path}

  tar -xzf ${tmp_path} -C ${unzip_path}
  mv "${unzip_path}/nvim-linux64/bin"   "${NEOVIM_INSTALL_PATH}/bin"
  mv "${unzip_path}/nvim-linux64/lib"   "${NEOVIM_INSTALL_PATH}/lib"
  mv "${unzip_path}/nvim-linux64/share" "${NEOVIM_INSTALL_PATH}/share"

  sudo rm $tmp_path
  sudo rm -r $unzip_path

	echo ' You should add '$HOME/.neovim/bin' to your $PATH and enable it use follow steps:'
	echo 'export PATH="$HOME/.neovim/bin:$PATH'
	echo 'source "$HOME/.zshrc"'
	return 0
}
