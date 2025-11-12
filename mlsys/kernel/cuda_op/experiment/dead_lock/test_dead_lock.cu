#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

namespace {
  // Undefined behavior. However, in general, people found __syncthreads won't
  // wait terminated threads, so this function won't hang in most cases.
  __global__ void dead_lock_sync_in_condition() {
    unsigned tid = threadIdx.x;
    if (tid % 2 == 0) {
      __syncthreads();
    }
  }

  __device__ int flag = 1;

  __global__ void syncthreads_dead_lock() {
    unsigned tid = threadIdx.x;
    if (threadIdx.x < 16) {
      printf("tid %d start\n", threadIdx.x);
      while (flag == 1) {
      }
    } else if (threadIdx.x < 32) {
      printf("tid %d start\n", threadIdx.x);
    } else {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
      flag = 0;
    }
  }

  __global__ void per_warp_per_syncthreads() {
    unsigned tid = threadIdx.x;
    if (threadIdx.x < 16) {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
      while (flag == 1) {
      }
    } else if (threadIdx.x < 32) {
      printf("tid %d start\n", threadIdx.x);
    } else {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
      flag = 0;
    }
  }

  __global__ void per_warp_two_syncthreads() {
    unsigned tid = threadIdx.x;
    if (threadIdx.x < 16) {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
      while (flag == 1) {
      }
    } else if (threadIdx.x < 32) {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
    } else {
      printf("tid %d start\n", threadIdx.x);
      __syncthreads();
      flag = 0;
    }
  }

} // namespace

int main() {
  // dead_lock_sync_in_condition<<<1, 1'024>>>();
  // syncthreads_dead_lock<<<1, 64>>>();
  per_warp_per_syncthreads<<<1, 64>>>();
  // per_warp_two_syncthreads<<<1, 64>>>();
  cudaDeviceSynchronize();
  return 0;
}

