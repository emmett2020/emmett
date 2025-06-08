Provide examples of cuda operator development combined with the C++23 standard, stdexec, msproxy and clangd.

# 1. Requirements
platform: ubuntu
g++ >= 14.2.0
nvcc >= 12.8

# 2. Strength
TODO

# 3. Build
```bash
cd /path/to/cuda_operators/
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j`nproc`
```

# 4. Run cases
```bash
./build/test/test_op
```


