#include <cstddef>
#include <iostream>
#include <cstdint>
#include <vector>
#include <random>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

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

  __global__ void fill_random_data_kernel(curandState* state, float* data, int n, uint64_t seed) {
    unsigned idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
      curand_init(seed, idx, 0, &state[idx]);  // 初始化随机数状态
      data[idx] = curand_uniform(&state[idx]); // 生成 [0, 1) 的随机数
    }
  }

  void throw_if(bool cond, std::string_view msg) {
    if (cond) {
      throw std::runtime_error{msg.data()};
    }
  }

  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  void cublas_check(cublasStatus_t err) {
    if (err != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error{cublasGetStatusString(err)};
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

  void cpu_mma(int M, int N, int K, const float* a_ptr, const float* b_ptr, float* c_ptr) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += a_ptr[(m * K) + k] * b_ptr[(n * K) + k];
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

  // const int K = cache_line_elements_cnt;
  const int K = 3;
  // const int M = cache_elements_cnt / K / 2;
  // const int N = cache_elements_cnt / K / 2;

  const int M = 2;
  const int N = 2;

  // const int num_tiles_m = 128;
  // const int num_tiles_n = 8;
  // const uint32_t tile_m = M / num_tiles_m;
  // const uint32_t tile_n = N / num_tiles_n;
  // auto grid_dim         = dim3{num_tiles_m, num_tiles_n};
  // auto block_dim        = dim3{tile_m};

  std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";
  // std::cout << "tile_m=" << tile_m << ", tile_n=" << tile_n << "\n";

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
  cpu_mma(M, N, K, a_cpu.data(), b_cpu.data(), c_cpu.data());

  // matmul_l2_cache_hit_rate_100<<<grid_dim, block_dim>>>(
  //   a_ptr,
  //   b_ptr,
  //   M,
  //   K,
  //   N,
  //   tile_m,
  //   tile_n,
  //   c_ptr);

  print_device_buffer(a_ptr, M, K, "a_ptr");
  print_device_buffer(b_ptr, N, K, "b_ptr");
  print_host_buffer(c_cpu.data(), M, N, "c_cpu_ptr");
  print_device_buffer(c_ptr, M, N, "c_dev_ptr");

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
}
