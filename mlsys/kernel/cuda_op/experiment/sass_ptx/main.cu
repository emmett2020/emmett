#include <cuda_runtime.h>

namespace {
  __global__ void Fill(float* A, float v) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    A[idx]       = v;
  }
} // namespace

int main() {
  return 0;
}
