#!/bin/bash
cat << END
https://www.anaconda.com/docs/getting-started/miniconda/install#linux
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              No              |
|------------------------------|------------------------------|
END
set -euo pipefail

arch=$(uname -m)

# Install
mkdir -p "${HOME}/miniconda3"
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${arch}.sh" -O "${HOME}/miniconda3/miniconda.sh"
bash "${HOME}/miniconda3/miniconda.sh" -b -u -p "${HOME}/miniconda3"
rm "${HOME}/miniconda3/miniconda.sh"

# Enable conda
source "${HOME}/miniconda3/bin/activate"
conda init --all
