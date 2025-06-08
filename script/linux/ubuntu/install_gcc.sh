#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
END

set -euo pipefail

CUR_SCRIPT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)

echo "Add universe warehose firstly"
bash "${CUR_SCRIPT_DIR}"/add_universe_warehose.sh

# Only support Ubuntu 24.04+ now.
. /etc/os-release
if [[ "${VERSION_ID%%.*}" -lt 24 ]]; then
  echo "Unsupported OS system version."
  exit 0
fi

echo "Install gcc-14"
sudo apt install -y gcc-14 g++-14 --fix-missing
gcc-14 --version
g++-14 --version

echo "Change it to default version"
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
