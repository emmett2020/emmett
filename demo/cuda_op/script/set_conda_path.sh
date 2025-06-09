#!/bin/bash

if [[ -z "${ENV_NAME}" ]]; then
  echo "Please provide ENV_NAME environment variable."
  exit 1
fi

# Get conda environment path.
env_path=$(conda env list | grep -E "\b${ENV_NAME}\s" | awk '{print $2}')
if [[ "${env_path}" == "*" ]]; then
  # Skip "*" if ENV_NAME is currently activating. e.g.:
  # cuda_op  * /path/to/miniconda3/envs/cuda_op
  env_path=$(conda env list | grep -E "\b${ENV_NAME}\s" | awk '{print $3}')
fi

# Use default path if still can't found the path of given env.
if [[ -z "$env_path" ]]; then
  echo "Try default conda path to get the path of conda ${ENV_NAME}."
  conda_base=$(conda info --base)
  default_path="${conda_base}/envs/${ENV_NAME}"
  if [[ -d "${default_path}" ]]; then
    env_path="${default_path}"
  fi
fi

if [[ -n "$env_path" ]]; then
  export PATH="${env_path}/bin:$PATH"
  echo "Found environment '${ENV_NAME}' at: ${env_path}"
else
  echo "conda environment '${ENV_NAME}' not found."
  exit 1
fi
