#!/bin/bash
cat << END
https://www.anaconda.com/docs/getting-started/miniconda/install#linux
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              No              |
|------------------------------|------------------------------|
END

# Install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Enable conda
source ~/miniconda3/bin/activate
conda init --all
