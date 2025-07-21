#include <cstddef>
#include <iostream>
#include <cstdint>
#include <vector>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "gemm_impl.cuh"

namespace {

  // thread block has two dimensions, thread only has one dimension
  // thread block y will split K
  __global__ void gemm_two_dims(
    const float* A,
    const float* B,
    int M,
    int K,
    int N,
    int tile_m,
    int tile_k,
    float* C) {
    unsigned a_block_row = blockIdx.y * tile_m;
    unsigned a_block_col = blockIdx.x * tile_k;
    unsigned a_row       = a_block_row + threadIdx.x;

    unsigned b_block_row = blockIdx.x * tile_k;

    for (int y = 0; y < N; ++y) {
      unsigned b_col = y;

      float sum = 0;
      for (int x = 0; x < tile_k; ++x) {
        unsigned a_col  = a_block_col + x;
        unsigned b_row  = b_block_row + x;
        sum            += A[(a_row * K) + a_col] * B[(b_row * N) + b_col];
      }
      float* ptr = C + (static_cast<size_t>(a_row * N)) + b_col;
      atomicAdd(ptr, sum);
    }
  }

  // thread block has two dimensions, thread also has two dimensions
  // thread block y will split K
  __global__ void gemm_two_dims_two_dims(
    const float* A,
    const float* B,
    int M,
    int K,
    int N,
    int tile_m,
    int tile_k,
    float* C) {
    unsigned a_block_row = blockIdx.y * tile_m;
    unsigned a_block_col = blockIdx.x * tile_k;
    unsigned a_row       = a_block_row + threadIdx.x;

    unsigned b_block_row = blockIdx.x * tile_k;

    for (int y = 0; y < N; ++y) {
      unsigned b_col = y;

      unsigned a_col = a_block_col + threadIdx.y;
      unsigned b_row = b_block_row + threadIdx.y;
      float sum      = A[(a_row * K) + a_col] * B[(b_row * N) + b_col];
      float* ptr     = C + (static_cast<size_t>(a_row * N)) + b_col;
      atomicAdd(ptr, sum);
    }
  }

  /// b is col_major
  void cpu_gemm(int M, int N, int K, const float* a_ptr, const float* b_ptr, float* c_ptr) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += a_ptr[(m * K) + k] * b_ptr[(k * N) + n];
        }
        c_ptr[(m * N) + n] = sum;
      }
    }
  }
} // namespace

auto main() noexcept(false) -> int {
  auto prop = cudaDeviceProp{};
  cuda_check(cudaGetDeviceProperties(&prop, 0)); // Query device 0
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
  // const int M = cache_elements_cnt / K / 2;
  // const int N = cache_elements_cnt / K / 2;

  const int M = 128;
  const int N = 32;
  std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";

  const auto a_num_eles = static_cast<int>(M * K);
  const auto b_num_eles = static_cast<int>(K * N);
  const auto c_num_eles = static_cast<int>(M * N);
  const size_t a_size   = static_cast<uint64_t>(M * K) * sizeof(float);
  const size_t b_size   = static_cast<uint64_t>(K * N) * sizeof(float);
  const size_t c_size   = static_cast<uint64_t>(M * N) * sizeof(float);
  float* a_ptr          = nullptr;
  float* b_ptr          = nullptr;
  float* c_ptr          = nullptr;
  cuda_check(cudaMalloc(&a_ptr, a_size));
  cuda_check(cudaMalloc(&b_ptr, b_size));
  cuda_check(cudaMalloc(&c_ptr, c_size));

  std::random_device rd{};
  fill_random_data(a_ptr, M * K, rd());
  fill_random_data(b_ptr, N * K, rd());

  std::vector<float> a_cpu(static_cast<size_t>(M * K), 0);
  std::vector<float> b_cpu(static_cast<size_t>(N * K), 0);
  std::vector<float> c_cpu(static_cast<size_t>(M * N), 0);
  cuda_check(cudaMemcpy(a_cpu.data(), a_ptr, a_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(b_cpu.data(), b_ptr, b_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cpu_gemm(M, N, K, a_cpu.data(), b_cpu.data(), c_cpu.data());


  // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100 * 1'024 * 1'024);

  launch_gemm_split_m_n_grid_1dim_blk_2dims(a_ptr, b_ptr, M, K, N, 16, c_ptr);

  print_device_buffer(a_ptr, M, K, "a_ptr");
  print_device_buffer(b_ptr, N, K, "b_ptr");
  print_host_buffer(c_cpu.data(), M, N, "c_cpu_ptr");
  print_device_buffer(c_ptr, M, N, "c_dev_ptr");

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
}
