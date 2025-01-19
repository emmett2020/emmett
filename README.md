# Introduction
This project mainly uses `C++`, and contains:

| directory | Notes                                                          |
| -------   | ------------------------------------------------               |
| demo      | Contains some small C++ projects                               |
| tutorial  | Contains some structured and educational code                  |
| bench     | Contains some benchmark code for C++                           |
| script    | Contains some useful scripts                                   |
| config    | Contains configs for zsh/neovim/docker and so on               |


# Usage
This repository contains several projects on independent directory. If you want to build one of them, just follow the tutorial below. If you want to use some projects on your own project, it is recommended to just copy files rather than reference this whole project.

## 1. Environment
Supported Platform: Most projects support on Linux platform while there still remains some projects only support MacOS.

Compiler: G++ and Clang++.

## 2. Build specific project
```shell
git clone --recursive https://github.com/emmett2020/emmett
cd the-project-directory-you-want-to-build
mkdir build && cd build
cmake -GNinja ..
ninja -j16 -v
```

## 3. The steps to add a new project
1. Create project in suitable directory.
2. Copy CMakeLists.txt to the new project. Then Add source files to CMakeLists.txt.
4. Add this new project into .clangd to enable LSP.
5. Add a README.md to describe what this new project will do.

