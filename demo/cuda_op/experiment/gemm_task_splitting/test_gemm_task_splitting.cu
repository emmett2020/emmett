#include <cstddef>
#include <iostream>
#include <cstdint>
#include <vector>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "gemm_impl.cuh"

namespace {
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
  const int K = 16;
  const int M = 128;
  const int N = 32;
  std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";

  const size_t a_size = static_cast<uint64_t>(M * K) * sizeof(float);
  const size_t b_size = static_cast<uint64_t>(K * N) * sizeof(float);
  const size_t c_size = static_cast<uint64_t>(M * N) * sizeof(float);
  float* a_ptr        = nullptr;
  float* b_ptr        = nullptr;
  float* c_ptr        = nullptr;
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

  launch_gemm_split_m_k_grid_2dim(a_ptr, b_ptr, M, K, N, 16, 4, c_ptr);

  print_device_buffer(a_ptr, M, K, "a_ptr");
  print_device_buffer(b_ptr, N, K, "b_ptr");
  print_host_buffer(c_cpu.data(), M, N, "c_cpu_ptr");
  print_device_buffer(c_ptr, M, N, "c_dev_ptr");

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
}
