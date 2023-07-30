# Introduction
Record some common code. Mainly contains:
- 😉 Some messy code `pieces`. This usually contains a single piece of knowledge or a simple use of a technology.
- 😃 Some fancy `demos`. This usually contains small items that combine many knowledge points.
- 😊 Some useful `tools`. This holds the source code for some utility tools I wrote.
- 🤔 Some useful `scripts`. `shell`, `python`, `lua` or other utility scripts are placed here.
- 🤩 Some structured and meaningful codes for `tutorial`. It is associated with the blog I wrote. It can be used directly anywhere.
- 😏 Some codes are stored for performance `pressure` measurement.
- 😛 Some universal `config` files. Includes `yaml`, `.zshrc` and other configuration files. The env configuration is also in there.
- 😎 Some `leetcode` solutions.
- ✨ Some really nice code, we put it in `thirdparties`.
- 😮 It contains `annotations` and analyses I've written for some of the impressive open source code.
- 😧 Some things which are `not_classified`.
- 😧 Some awesome library and framework are placed int `thirdparties`.
 
This project mainly uses `C++`, but will also use other languages and even include some `frontend` and `client` content. 
Each folder contains the relevant environment dependencies and sub-project descriptions.

# Usage
When using this project, it is recommended to copy files from a certain directory to the newly created project, rather than directly reference the complete project. 

## 1. Environment

| Type           | Value1        | Value2         |
| :---           | :----:        | :----:         |
| Platform       | Macos         |  Ubuntu        |
| Compiler       | g++(12.0+)    |  clang++       |

Note: Some sub-projects require specific environmental dependencies. Details of such sub-projects are described in their `README.md`.

## 2. Build project
```shell
mkdir build && cd build

# Replace BUILD_XXX to the project you want to build.
cmake -DBUILD_XXX .. 

# Ninja is suggest to build specific sub-project.
#cmake -G ninja -DBUILD_XXX ..

./path_to_bin/xxx
```
Some sub-projects require specific environmental dependencies. Such sub-projects are not compiled by default, and a new flag is required to compile them.


# TODO
- [ ] add pre-commit
- [ ] add clang-tidy check to pre-commit
- [ ] repair clangtidy warning
- [ ] provide a script file to fast cleanup the codebase. such as: remove all **/build/.
- [ ] transfer todo list to issues in Github.
- [ ] Some projects have dependent non-code files. You should copy it to the execution directory under `build`. Otherwise, every time you change the non-code file to test demos, it is equivalent to changing the original code.