#!/bin/bash

# TODO: Support zsh shell.

set -euo pipefail

CUR_SCRIPT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)

# Install miniconda if local environment doesn't have one.
if ! command -v conda &> /dev/null; then
  echo "Error: Conda not found. Please ensure Conda is installed and in your PATH."
  exit 1
fi

# Prepare cuda_op virtual environment.
ENV_NAME="cuda_op"
if conda env list | grep -qE "\b${ENV_NAME}\b"; then
  echo "Environment '${ENV_NAME}' exists. Activating..."
  conda activate "${ENV_NAME}"
else
  echo "Environment '${ENV_NAME}' not found. Creating..."
  conda create -n "${ENV_NAME}" python==3.11 -y
  echo "Activating new environment '${ENV_NAME}'..."
  conda activate "${ENV_NAME}"
fi

# Set necessary conda environment variables.
source "${CUR_SCRIPT_DIR}/script/set_conda_path.sh"

# Set nvcc and cuda package.
if ! command -v nvcc &> /dev/null; then
  default_cuda_path="/usr/local/cuda/bin/"
  echo "WARN: The nvcc compiler not found, try default cuda path: ${default_cuda_path}"
  export PATH="${PATH}:${default_cuda_path}"
  if ! command -v nvcc &> /dev/null; then
    echo "ERROR: The nvcc compiler not found."
    exit 1
  fi
fi
