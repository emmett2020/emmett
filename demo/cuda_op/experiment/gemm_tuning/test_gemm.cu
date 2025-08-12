#include <cstddef>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "gemm_ampere_simt.cuh"
#include "gemm_ampere_tcore.cuh"
#include "gemm_cutlass_simt.cuh"

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

  void valid(const float* golden, const float* output, int M, int N) {
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float g = golden[m * N + n];
        float o = output[m * N + n];
        if (std::fabs(g - o) >= atol + rtol * std::fabs(o)) {
          printf("m=%d, n=%d, g=%f != o=%f\n", m, n, g, o);
          throw std::runtime_error{"mismatch"};
        }
      }
    }
  }
} // namespace

auto main(int argc, char** argv) noexcept(false) -> int {
  int M = -1;
  int N = -1;
  int K = -1;
  if (argc != 4) {
    M = 1'024;
    N = 1'024;
    K = 1'024;
  } else {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\n";

  const size_t a_size  = static_cast<uint64_t>(M * K) * sizeof(float);
  const size_t b_size  = static_cast<uint64_t>(K * N) * sizeof(float);
  const size_t c_size  = static_cast<uint64_t>(M * N) * sizeof(float);
  float* a_ptr         = nullptr;
  float* b_ptr         = nullptr;
  float* c_ptr         = nullptr;
  float* c_cutlass_ptr = nullptr;
  cuda_check(cudaMalloc(&a_ptr, a_size));
  cuda_check(cudaMalloc(&b_ptr, b_size));
  cuda_check(cudaMalloc(&c_ptr, c_size));
  cuda_check(cudaMalloc(&c_cutlass_ptr, c_size));

  std::random_device rd{};
  fill_random_data(a_ptr, M * K, rd());
  fill_random_data(b_ptr, N * K, rd());

  std::vector<float> a_cpu(static_cast<size_t>(M * K), 0);
  std::vector<float> b_cpu(static_cast<size_t>(N * K), 0);
  std::vector<float> c_cpu(static_cast<size_t>(M * N), 0);
  cuda_check(cudaMemcpy(a_cpu.data(), a_ptr, a_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(b_cpu.data(), b_ptr, b_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cpu_gemm(M, N, K, a_cpu.data(), b_cpu.data(), c_cpu.data());
  launch_gemm(a_ptr, b_ptr, M, N, K, c_ptr);
  CutlassGemmAmpere(a_ptr, b_ptr, M, N, K, c_cutlass_ptr);

  std::vector<float> cuda_data(M * N);
  std::vector<float> cutlass_data(M * N);
  cuda_check(cudaMemcpy(cuda_data.data(), c_ptr, M * N * 4, cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(cutlass_data.data(), c_cutlass_ptr, M * N * 4, cudaMemcpyDeviceToHost));
  std::cout << "Comparing cuda with cpu" << "\n";
  valid(c_cpu.data(), cuda_data.data(), M, N);
  std::cout << "Comparing cutlass with cpu" << "\n";
  valid(c_cpu.data(), cutlass_data.data(), M, N);
  std::cout << "Run passed" << "\n";

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_ptr));
  cuda_check(cudaFree(c_cutlass_ptr));
}
