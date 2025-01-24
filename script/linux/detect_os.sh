#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              No              |
|------------------------------|------------------------------|
END

OS=$(uname | tr '[:upper:]' '[:lower:]')

detect_os() {
  if [ "${OS}" = "linux" ]; then
    # Check for specific Linux flavors
    if [ -f /etc/lsb-release ]; then
      # For some versions of Ubuntu and Linux Mint
      . /etc/lsb-release
      OS=${DISTRIB_ID}
    elif [ -f /etc/debian_version ]; then
      # Older Debian, Ubuntu, etc.
      OS="Debian"
    elif [ -f /etc/redhat-release ]; then
      # Redhat/CentOS
      OS="CentOS"
    elif [ -f /etc/os-release ]; then
      # Modern systems
      . /etc/os-release
      OS=$ID
    fi
  elif [ "$OS" = "darwin" ]; then
    OS="MacOS"
  else
    OS="Unknown"
  fi

  echo "Current operating system: ${OS}"
}

detect_os
