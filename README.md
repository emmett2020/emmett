# Emmett
![License](https://img.shields.io/github/license/emmett2020/emmett)

| CI                         |
| -------------------        |
| ![nightly build docker](https://github.com/emmett2020/emmett/actions/workflows/nightly_build_docker.yml/badge.svg)       |
| ![build ubuntu script](https://github.com/emmett2020/emmett/actions/workflows/ci_ubuntu_scripts.yml/badge.svg)        |
| ![build docker daily](https://github.com/emmett2020/emmett/actions/workflows/ci_build_docker_daily.yml/badge.svg)         |
| ![demo](https://github.com/emmett2020/emmett/actions/workflows/ci_cpp.yml/badge.svg)                       |
| ![check typo](https://github.com/emmett2020/emmett/actions/workflows/ci_check_typo.yml/badge.svg)                 |
| ![tutorial](https://github.com/emmett2020/emmett/actions/workflows/ci_tutorial.yml/badge.svg)                   |


# Introduction

This project is primarily for personal use, including small C++ projects I've written, projects in the MLSYS field, as well as configurations for various tools and scripts for specific tasks. All content is validated by GitHub CI. Issues and PRs are welcome!

**NOTE**:

1. This repository is primarily for **personal use**, and any content may be subject to change in the future. However, the repository will still strive to adhere to mainstream version release rules as much as possible.
2. Please retain the copyright notice when using the content from this repository.

Below is a detailed introduction to subdirectories:

| Directory | Explanation                                                    |
| -------   | ------------------------------------------------               |
| cpp       | Contains several C/C++ projects I've written                   |
| mlsys     | Contains several mlsys projects                                |
| cmake     | Contains CMake utility                                         |
| tutorial  | Contains some structured and educational code                  |
| script    | Contains some platform specific scripts I wrote                |
| config    | Contains configurations for some popular tools I used          |


# How to build subprojects
Each subproject may have its own environmental dependency requirements. Generally, a Linux platform and a compiler supporting C++20 or later are required. Specific environmental configuration requirements depend on the subproject. Some more complex subprojects may provide documentation for environment setup scripts.

Once the environment meets the requirements, the typical compilation process is as follows:
```bash
git clone --recursive https://github.com/emmett2020/emmett
cd /path/to/subproject/you/want/to/build
mkdir build && cd build
cmake ..
make -j`nproc`
```

# How to contribution to this project
[contribute](./doc/Contribution.md)

# What will support in the future
[TODOs](./doc/TODOs.md)


