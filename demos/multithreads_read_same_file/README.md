# Dependencies
- Ubuntu
- boost 1.83
- CMake 3.29

# Build
```bash
# Install boost into system
bash emmett/scripts/install_boost.sh

mkdir build && cd build
cmake -GNinja ..
ninja -j16
```

