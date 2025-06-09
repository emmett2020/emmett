Provide high performance implementation of some cuda operators.

# 1. Requirements
platform: ubuntu
g++ >= 14.2.0
nvcc >= 12.8

# 2. Build
```bash
conda create -n cuda_op python==3.11                                                                             ─
conda activate cuda_op
cd /path/to/cuda_op/
pip3 install .
```

# 3. Run cases
```bash
./build/test/test_op
```


