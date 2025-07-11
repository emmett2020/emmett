#include <iostream>
#include "thread_block_swizzle/sequential.cuh"

auto main() -> int {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0); // Query device 0
  double cache_size          = prop.l2CacheSize / 1'024.0 / 1'024.0;
  double persisting_max_size = prop.persistingL2CacheMaxSize / 1'024.0 / 1'024.0;

  std::cout << "L2 Cache size: " << cache_size << "MB \n";
  std::cout << "Persisting L2 Cache max size: " << persisting_max_size << "MB \n";
}
