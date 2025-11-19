#include <iostream>
#include <stdint.h>

#include "thread_block_swizzle/sequential.cuh"

namespace {
  __global__ void matmul_l2_cache_hit_rate_100(
    const float* a_ptr,
    const float* b_ptr,
    int M,
    int K,
    int N,
    int tile_m,
    int tile_n,
    float* c_ptr) {
    unsigned a_tile_size   = tile_m * K;
    unsigned b_tile_size   = K * tile_n;
    unsigned c_tile_size   = tile_m * tile_n;
    unsigned a_tile_offset = (blockIdx.x * gridDim.y * a_tile_size) + (blockIdx.y * a_tile_size);
    unsigned b_tile_offset = (blockIdx.x * gridDim.y * b_tile_size) + (blockIdx.y * b_tile_size);
    unsigned c_tile_offset = (blockIdx.x * gridDim.y * c_tile_size) + (blockIdx.y * c_tile_size);

    unsigned tx = threadIdx.x;

    for (int y = 0; y < tile_n; ++y) {
      float sum = 0;
      for (int x = 0; x < K; ++x) {
        float a  = a_ptr[a_tile_offset + (tx * K) + x];
        float b  = b_ptr[b_tile_offset + (tx * K) + x];
        sum     += a * b;
      }
      c_ptr[c_tile_offset + (tx * tile_n) + y] = sum;
    }
  }

  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  void print_matrix();
} // namespace

auto main() -> int {
  auto prop = cudaDeviceProp{};
  cudaGetDeviceProperties(&prop, 0); // Query device 0
  double cache_size          = prop.l2CacheSize / 1'024.0 / 1'024.0;
  double persisting_max_size = prop.persistingL2CacheMaxSize / 1'024.0 / 1'024.0;

  std::cout << "L2 Cache size: " << cache_size << "MB \n";
  std::cout << "Persisting L2 Cache max size: " << persisting_max_size << "MB \n";

  const int cache_line_byte_size    = 128;
  const int cache_line_elements_cnt = cache_line_byte_size / sizeof(float);
  const int cache_elements_cnt      = std::ceil(prop.l2CacheSize / sizeof(float));
  std::cout
    << "cache_elements_cnt: "
    << cache_elements_cnt
    << ", cache_line_elements_cnt: "
    << cache_line_elements_cnt
    << "\n";

  const int K = cache_line_elements_cnt;
  const int M = cache_elements_cnt / K / 2;
  const int N = cache_elements_cnt / K / 2;

  const int num_tiles_m = 128;
  const int num_tiles_n = 8;
  const uint32_t tile_m = M / num_tiles_m;
  const uint32_t tile_n = N / num_tiles_n;
  auto grid_dim         = dim3{num_tiles_m, num_tiles_n};
  auto block_dim        = dim3{tile_m};

  std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";
  std::cout << "tile_m=" << tile_m << ", tile_n=" << tile_n << "\n";

  const size_t a_size = M * K * sizeof(float);
  const size_t b_size = K * N * sizeof(float);
  const size_t c_size = M * N * sizeof(float);
  float* a_ptr        = nullptr;
  float* b_ptr        = nullptr;
  float* c_ptr        = nullptr;
  cuda_check(cudaMalloc(&a_ptr, a_size));
  cuda_check(cudaMalloc(&b_ptr, b_size));
  cuda_check(cudaMalloc(&c_ptr, c_size));

  // matmul_l2_cache_hit_rate_100<<<grid_dim, block_dim>>>(
  //   a_ptr,
  //   b_ptr,
  //   M,
  //   K,
  //   N,
  //   tile_m,
  //   tile_n,
  //   c_ptr);

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
}
