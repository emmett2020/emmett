#!/bin/bash
set -e

# 1. /path-to-emmett/configs/zshrc/daily must be existed.
# 2. Could be used both on Ubuntu and MacOS

link_oh_my_zsh=https://install.ohmyz.sh/
link_zsh_syntax_highlighting=https://github.com/zsh-users/zsh-syntax-highlighting
auto_suggestions_link=https://github.com/zsh-users/zsh-autosuggestions
link_powerlevel10k=https://gitee.com/romkatv/powerlevel10k.git
link_eza="https://github.com/eza-community/eza/releases/download/v0.19.4/eza_x86_64-unknown-linux-gnu.tar.gz"
chroma_link="https://github.com/alecthomas/chroma/releases/download/v2.14.0/chroma-2.14.0-linux-amd64.tar.gz"

cur_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
omz_custom_dir="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

dir_oh_my_zsh="${HOME}/.oh-my-zsh"
install_dir="${omz_custom_dir}/plugins/zsh-syntax-highlighting"
dir_zsh_auto_suggesstions="${omz_custom_dir}/plugins/zsh-autosuggestions"
powerlevel10k_install_dir="${omz_custom_dir}/themes/powerlevel10k"

emmett_path="${cur_dir}/.."
zshrc_path="${emmett_path}/configs/zshrc/daily"

function get_system_info() {
	if grep -q "Ubuntu" "/etc/os-release" ; then
	  SYSTEM_TYPE="Ubuntu"
	  SYSTEM_VERSION=$(cat /etc/os-release | grep VERSION_ID | awk -F\" '{print $2}')
	  SYSTEM_ARCH=$(uname -m)
	fi
}


function check_zshrc() {
  if [[ ! -f "${zshrc_path}" ]]; then
    echo "  Can't find .zshrc file in emmett2020/emmett repo. Path of .zshrc: ${zshrc_path}"
    exit 1
  fi
  if [[ -f "${HOME}/.zshrc" ]]; then
    echo "  Remove or save ${HOME}/.zshrc first."
    exit 1
  fi
}

function print_hint() {
  echo -e "  ......\n\n"
  echo "  Installed zsh, config and plugins(needs sudo permission)"
  echo "  oh-my-zsh download  link: ${link_oh_my_zsh}"
  echo "  oh-my-zsh installed path: ${dir_oh_my_zsh}"
  echo "  zsh-syntax-highlighting download  link: ${link_zsh_syntax_highlighting}"
  echo "  zsh-syntax-highlighting installed path: ${install_dir}"
  echo "  zsh-autosuggestions download  link: ${auto_suggestions_link}"
  echo "  zsh-autosuggestions installed path: ${dir_zsh_auto_suggesstions}"
  echo "  powerlevel10k download  link: ${link_powerlevel10k}"
  echo "  powerlevel10k installed path: ${powerlevel10k_install_dir}"
  echo "  eza download link: ${link_eza}"
  echo "  chroma download link: ${chroma_link}"
  echo "  Successfully installed zsh, omz, themes, config and plugins."
  echo "  Please use: source ~/.zshrc to apply newest config."
}

function install_omz() {
	local tmp="${HOME}/.tmp_install"
  [[ -d "${tmp}" ]] && rm -r "${tmp}"
  mkdir -p "${tmp}"

  [[ -d "${dir_oh_my_zsh}" ]]               && rm -r "${dir_oh_my_zsh}"
  [[ -d "${powerlevel10k_install_dir}" ]]           && rm -r "${powerlevel10k_install_dir}"
  [[ -d "${dir_zsh_auto_suggesstions}" ]]   && rm -r "${dir_zsh_auto_suggesstions}"
  [[ -d "${install_dir}" ]] && rm -r "${install_dir}"

  source "${cur_dir}/detect_os.sh"
  if [[ "${OS}" == "MacOS" ]]; then
    brew install zsh chroma eva
  elif [[ "${OS}" == "Ubuntu" ]]; then
    sudo apt install -y zsh

    wget "${link_eza}" -O "${tmp}/eva.tar.gz"
    tar -xzf "${tmp}/eva.tar.gz" -C "${tmp}"
    mv "${tmp}/eza" "/usr/local/bin"

    wget "${chroma_link}" -O "${tmp}/chroma.tar.gz"
    tar -xzf "${tmp}/chroma.tar.gz" -C "${tmp}"
    mv "${tmp}/chroma" "/usr/local/bin"
  else
    exit 1
  fi

  # 2. Install oh my zsh
  wget "${link_oh_my_zsh}" -O "${tmp}/oh_my_zsh.sh"
  bash "${tmp}/oh_my_zsh.sh" --unattended

  # 3. Install plugins and themes
  git clone "${link_zsh_syntax_highlighting}" "${install_dir}"
  git clone "${auto_suggestions_link}"    "${dir_zsh_auto_suggesstions}"
  git clone --depth=1 "${link_powerlevel10k}" "${powerlevel10k_install_dir}"

  # 6. Copy zshrc from emmett repo
  cp "${zshrc_path}" ~/.zshrc

  rm -rf ${tmp}
}

check_zshrc
install_omz
print_hint
