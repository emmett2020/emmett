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

echo "Install gcc-14"
sudo apt install -y gcc-14 g++-14 --fix-missing
gcc-14 --version
g++-14 --version

echo "Change it to default version"
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
