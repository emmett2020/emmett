#!/bin/bash
cat << END
|------------------------------|------------------------------|
|          item                |         explanation          |
|------------------------------|------------------------------|
|    needs root permission?    |              Yes             |
|------------------------------|------------------------------|
Universe (Community Maintained Warehouse)
END

set -euo pipefail
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update
