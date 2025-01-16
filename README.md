# Introduction
This project mainly uses `C++`, and contains:
- 😃 Some fancy `demos`.
- 🤔 Some useful `scripts`.
- 🤩 Some structured and meaningful codes for `tutorial`. It is associated with the blog I wrote.
- 😏 Some codes are stored in `bench` for performance measurement.
- 😛 Some universal `config` files. Includes `yaml`, `.zshrc` and other configuration files.

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



# TODO
- [ ] Add pre-commit
- [ ] Add clang-tidy and repair clang-tidy warnings
- [ ] Provide a script file to fast cleanup the codebase. such as: remove all **/build/.
- [ ] Transfer TODOs in source code to Github issues.
