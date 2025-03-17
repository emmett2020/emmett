#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
END

set -euo pipefail

apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-venv
