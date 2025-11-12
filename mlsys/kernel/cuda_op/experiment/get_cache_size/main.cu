#include <iostream>
#include <cuda_runtime.h>

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  printf("Shared Memory per SM: %f KB\n", prop.sharedMemPerMultiprocessor / 1'024.0F);
  printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1'024);
}
