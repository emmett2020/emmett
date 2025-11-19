# Introduction

This project demonstrates how to configure a CUDA project using CMake and the [clangd](https://clangd.llvm.org/) language server.

## Prerequisites

* The [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) must be installed. Verify your installation by running `nvcc --version`.
* **Note:** If multiple CUDA versions are installed, this CMake configuration will prefer the one associated with the `nvcc` in your path.

## Building the Project

To build the project, run the following commands:

```bash
mkdir build && cd build
cmake ..
make -j`nproc`
```


## Configuring `clangd`

For accurate code completion and hints from `clangd`, you may need to adjust the `--cuda-path` setting in the `.clangd` configuration file. This is only necessary if your `CUDA` installation is not located at the default path `/usr/local/cuda`.
