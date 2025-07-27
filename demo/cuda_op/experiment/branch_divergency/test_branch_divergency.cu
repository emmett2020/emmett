#include <cstdio>
#include <cuda_runtime.h>

namespace {
  __global__ void branch_divergence() {
    if (threadIdx.x % 2 == 0) {
      printf("threadIdx.x %u, executes if \n", threadIdx.x);
    } else {
      printf("threadIdx.x %u, executes else \n", threadIdx.x);
    }
  }

  // TODO: Example
  __global__ void branch_divergence_independet_thread_scheduling() {
    if (threadIdx.x % 2 == 0) {
      if (threadIdx.x > 4) {
        printf("threadIdx.x %u, executes if if \n", threadIdx.x);
      } else {
        printf("threadIdx.x %u, executes if else \n", threadIdx.x);
      }
    } else {
      if (threadIdx.x > 4) {
        printf("threadIdx.x %u, executes else if \n", threadIdx.x);
      } else {
        printf("threadIdx.x %u, executes else else \n", threadIdx.x);
      }
    }
  }


} // namespace

int main() {
  // branch_divergence<<<1, 16>>>();
  branch_divergence_independet_thread_scheduling<<<1, 16>>>();
  cudaDeviceSynchronize();
  return 0;
}

