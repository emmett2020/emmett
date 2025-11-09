# Emmett
![License](https://img.shields.io/github/license/emmett2020/emmett)

| CI                         |
| -------------------        |
| ![nightly build docker](https://github.com/emmett2020/emmett/actions/workflows/nightly_build_docker.yml/badge.svg)       |
| ![build ubuntu script](https://github.com/emmett2020/emmett/actions/workflows/ci_ubuntu_scripts.yml/badge.svg)        |
| ![build docker daily](https://github.com/emmett2020/emmett/actions/workflows/ci_build_docker_daily.yml/badge.svg)         |
| ![bench](https://github.com/emmett2020/emmett/actions/workflows/ci_bench.yml/badge.svg)                      |
| ![demo](https://github.com/emmett2020/emmett/actions/workflows/ci_demo.yml/badge.svg)                       |
| ![check typo](https://github.com/emmett2020/emmett/actions/workflows/ci_check_typo.yml/badge.svg)                 |
| ![tutorial](https://github.com/emmett2020/emmett/actions/workflows/ci_tutorial.yml/badge.svg)                   |


A suite of tested C/C++/CUDA/CMake projects and cross-platform scripts (Shell/Lua/Python), complete with development configurations. All content is validated by GitHub CI. Issues and PRs are welcome!

Note: Please retain the copyright notice when using the content from this repository.

Below is a detailed introduction to subdirectories:

| directory | Notes                                                          |
| -------   | ------------------------------------------------               |
| cpp       | Contains several C/C++ projects                                |
| cuda      | Contains several CUDA projects                                 |
| cmake     | Contains CMake utility                                         |
| tutorial  | Contains some structured and educational code                  |
| script    | Contains some platform specific scripts                        |
| config    | Contains configs for zsh/neovim/docker and so on               |


# How to build subprojects
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
[contribute](./doc/Contribution.md)

# What will support in the future
[TODOs](./doc/TODOs.md)


