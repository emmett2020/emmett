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

# Catalog
- [Introduction](#Introduction)
- [Development environment configuration](#Development-environment-configuration)
- [Build sub-projects](#Build-sub-projects)
- [Contribution](#Contribution)
- [TODO](#TODO)

[toc]

# Introduction

Hello experts, welcome to my personal project👋.

This project is primarily for **personal use**, including small C++ projects I've written, projects in the MLSYS field, as well as configurations for various tools and scripts for specific tasks. All content is validated by GitHub CI. Issues and PRs are welcome!

**NOTE**:

1. Any content may be subject to change in the future. However, the repository will still strive to adhere to mainstream version release rules as much as possible.
2. Please retain the copyright notice when using the content from this repository.

Below is a detailed introduction to subdirectories:

| Directory | Explanation                                                    |
| -------   | ------------------------------------------------               |
| cpp       | Contains several C/C++ projects                                |
| mlsys     | Contains several mlsys projects                                |
| cmake     | Contains CMake utility                                         |
| tutorial  | Contains some structured and educational code                  |
| script    | Contains some platform specific scripts I wrote                |
| config    | Contains configurations for some popular tools I used          |

# Development environment configuration

This project is organized as a collection of subprojects, each designed for a specific purpose and may have its own dependencies. Generally, a `Linux`` platform and a compiler supporting `C++20` or later are required. Specific environmental configuration requirements depend on the subproject. Some more complex subprojects may provide documentations or scripts for environment setup. Despite this modularity, most sub-projects [use this development environment](./doc/config_ubuntu_develop_environment.md). You should prioritize configuring the environment and then compile the subprojects.

# Build sub-projects

Once the environment meets the requirements, the **typical** compilation process is as follows:
```bash
git clone --recursive https://github.com/emmett2020/emmett
cd /path/to/subproject/you/want/to/build
mkdir build && cd build
cmake ..
make -j`nproc`
```

# Contribution
[contribution](./doc/contribution.md)

# TODO
[TODOs](./doc/todo.md)


