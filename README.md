# Introduction
Mainly contains:
- 😉 Some messy code `pieces`.
- 😃 Some fancy `demos`.
- 😊 Some useful `tools`. This holds the source code for some utility tools.
- 🤔 Some useful `scripts`. `shell`, `python`, `lua` or other utility scripts are placed here.
- 🤩 Some structured and meaningful codes for `tutorial`. It is associated with the blog I wrote. It can be used directly anywhere.
- 😏 Some codes are stored in `bench` for performance measurement.
- 😛 Some universal `config` files. Includes `yaml`, `.zshrc` and other configuration files.
- ✨ Some really nice code, we put it in `thirdparties`.
- 😮 It contains `annotations` and analyses I've written for some of the impressive open source code.
- 😧 Some things which are `not_classified`.
 
This project mainly uses `C++`, but will also use other languages and even include some `frontend` and `client` content.

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
ninja -j16
```

## 3. The steps to add a new project
1. Create project directory.
2. Copy CMakeLists.txt to the new directory
3. Refine this file.
4. Add this directory to .clangd to enable LSP.
5. Coding



# TODO
- [ ] Add pre-commit
- [ ] Add clang-tidy and repair clang-tidy warnings
- [ ] Provide a script file to fast cleanup the codebase. such as: remove all **/build/.
- [ ] Transfer TODOs in source code to Github issues.
