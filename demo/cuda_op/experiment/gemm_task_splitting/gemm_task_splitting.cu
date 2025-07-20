#include <cstddef>
#include <iostream>
#include <cstdint>
#include <vector>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "gemm_split_m.cuh"

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

  __global__ void
  gemm(const float* A, const float* B, int M, int K, int N, int tile_size, float* C) {
    __shared__ float As[64][64];
    __shared__ float Bs[64][64];

    unsigned block_row = blockIdx.y;
    unsigned block_col = blockIdx.x;
    unsigned tx        = threadIdx.x;
    unsigned ty        = threadIdx.y;

    unsigned row = (block_row * tile_size) + ty;
    unsigned col = (block_col * tile_size) + tx;

    float sum = 0.0F;

    for (unsigned t = 0; t < K; t += tile_size) {
      unsigned a_col = t + tx;

      printf("datad %d %d %d %d, row=%d, col=%d\n",
             blockIdx.y,
             blockIdx.x,
             threadIdx.x,
             threadIdx.y,
             row,
             a_col);

      // Load A
      if (row < M && a_col < K) {
        As[ty][tx] = A[(row * K) + a_col];
      } else {
        As[ty][tx] = 0.0F;
      }

      // Load B
      unsigned b_row = t + ty;
      if (b_row < K && col < N) {
        Bs[ty][tx] = B[(b_row * N) + col];
      } else {
        Bs[ty][tx] = 0.0F;
      }
      __syncthreads();

      for (int k = 0; k < tile_size; k++) {
        if (t + k < K) {
          sum += As[ty][k] * Bs[k][tx];
        }
      }

      __syncthreads();
    }

    if (row < M && col < N) {
      C[(row * N) + col] = sum;
    }
  }

  __global__ void simple_matmul(const float* A, const float* B, int M, int K, int N, float* C) {
    unsigned row = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < M && col < N) {
      float sum = 0.0F;

      for (int k = 0; k < K; k++) {
        sum += A[(row * K) + k] * B[(k * N) + col];
      }

      C[(row * N) + col] = sum;
    }
  }

  __global__ void fill_random_data_kernel(curandState* state, float* data, int n, uint64_t seed) {
    unsigned idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
      curand_init(seed, idx, 0, &state[idx]);
      data[idx] = curand_uniform(&state[idx]);
    }
  }

  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  void fill_random_data(float* d_data, int num_elements, uint64_t seed) {
    curandState* d_state = nullptr;

    cuda_check(cudaMalloc(&d_state, num_elements * sizeof(curandState)));

    int block_size = 256;
    int grid_size  = (num_elements + block_size - 1) / block_size;
    fill_random_data_kernel<<<grid_size, block_size>>>(d_state, d_data, num_elements, seed);
  }

  void print_host_buffer(const float* h_data, int H, int W, std::string_view title = "") {
    throw_if(H <= 0, "H must be greater than 0");
    throw_if(W <= 0, "W must be greater than 0");
    int num_elements = H * W;
    size_t byte_size = num_elements * sizeof(float);
    if (!title.empty()) {
      std::cout << title << "\n";
    }
    for (int h = 0; h < H; h++) {
      std::cout << "h=" << h << "\t";
      for (int w = 0; w < W - 1; ++w) {
        std::cout << h_data[(h * W) + w] << ", ";
      }
      std::cout << h_data[(h * W) + W - 1] << "\n";
    }
  }

  void print_device_buffer(const float* d_data, int H, int W, std::string_view title = "") {
    throw_if(H <= 0, "H must be greater than 0");
    throw_if(W <= 0, "W must be greater than 0");
    int num_elements = H * W;
    size_t byte_size = num_elements * sizeof(float);
    std::vector<float> h_data(num_elements);
    cuda_check(cudaMemcpy(h_data.data(), d_data, byte_size, cudaMemcpyDeviceToHost));
    print_host_buffer(h_data.data(), H, W, title);
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

  launch_gemm_split_m_grid_2dims_blk_2dims_shared(a_ptr, b_ptr, M, K, N, 16, 8, c_ptr);

  print_device_buffer(a_ptr, M, K, "a_ptr");
  print_device_buffer(b_ptr, N, K, "b_ptr");
  print_host_buffer(c_cpu.data(), M, N, "c_cpu_ptr");
  print_device_buffer(c_ptr, M, N, "c_dev_ptr");

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
}
