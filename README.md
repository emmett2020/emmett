# Emmett
![License](https://img.shields.io/github/license/emmett2020/emmett)

| CI                         |
| -------------------        |
| ![nightly build docker daily](https://github.com/emmett2020/emmett/actions/workflows/nightly_build_docker_daily.yml/badge.svg) |
| ![build ubuntu script](https://github.com/emmett2020/emmett/actions/workflows/build_ubuntu_scripts.yml/badge.svg)        |
| ![build daily docker](https://github.com/emmett2020/emmett/actions/workflows/build_daily_docker.yml/badge.svg)         |
| ![bench](https://github.com/emmett2020/emmett/actions/workflows/ci_bench.yml/badge.svg)                      |
| ![demo](https://github.com/emmett2020/emmett/actions/workflows/ci_demo.yml/badge.svg)                       |
| ![tutorial](https://github.com/emmett2020/emmett/actions/workflows/ci_tutorial.yml/badge.svg)                   |


This repository contains code examples for C/C++/CMake, commonly used scripts for various platforms, development environment configuration files, and more. The relevant content has been tested as much as possible via GitHub CI and can be used directly. If you have any good ideas, please feel free to submit issues and PRs.

This repository is primarily for personal use, and any content may be subject to change in the future. However, the repository will still strive to adhere to mainstream version release rules as much as possible.

Note: Please retain the copyright notice when using the content from this repository.

Below is a detailed introduction to some subdirectories:

| directory | Notes                                                          |
| -------   | ------------------------------------------------               |
| demo      | Contains some small C++ projects                               |
| tutorial  | Contains some structured and educational code                  |
| bench     | Contains some benchmark code for C++                           |
| script    | Contains some useful scripts                                   |
| config    | Contains configs for zsh/neovim/docker and so on               |
| cmake     | Contains CMake files                                           |


# How to build C/C++ Projects
Each subproject will have its own environmental dependency requirements. Generally, a Linux platform and a compiler supporting C++20 or later are required. Specific environmental configuration requirements depend on the subproject. Some more complex subprojects may provide documentation for environment setup scripts.

Once the environment meets the requirements, the typical compilation process is as follows:
```bash
git clone --recursive https://github.com/emmett2020/emmett
cd /path/to/subproject/you/want/to/build
mkdir build && cd build
cmake ..
make -j`nproc`
```

# How to contribution to this project
[contribute](./docs/Contribution.md)

# What will support in the future
[TODOs](./docs/TODOs.md)


