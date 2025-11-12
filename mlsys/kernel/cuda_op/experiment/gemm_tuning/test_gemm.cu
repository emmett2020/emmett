#include <cstddef>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

#include "gemm_base.cuh"
#include "ampere/tcore_base.cuh"
#include "ampere/tcore_smem.cuh"
#include "gemm_cutlass_tcore.cuh"

namespace {
  void cpu_gemm(int M, int N, int K, const half* a_ptr, const half* b_ptr, half* c_ptr) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
          sum += static_cast<float>(a_ptr[(m * K) + k]) * static_cast<float>(b_ptr[(k * N) + n]);
        }
        c_ptr[(m * N) + n] = sum;
      }
    }
  }

  void valid(const half* golden, const half* output, int M, int N) {
    const float atol = 0.001;
    const float rtol = 0.024;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float g = golden[m * N + n];
        float o = output[m * N + n];
        if (std::fabs(g - o) >= atol + rtol * std::fabs(o)) {
          float error = std::fabs(g - o) / std::fabs(o);
          printf("m=%d, n=%d, g=%f != o=%f rerror=%f\n", m, n, g, o, error);
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

  const size_t a_size = static_cast<uint64_t>(M * K) * sizeof(half);
  const size_t b_size = static_cast<uint64_t>(K * N) * sizeof(half);
  const size_t c_size = static_cast<uint64_t>(M * N) * sizeof(half);
  half* a_ptr         = nullptr;
  half* b_ptr         = nullptr;
  half* c_cuda_ptr    = nullptr;
  half* c_cutlass_ptr = nullptr;
  cuda_check(cudaMalloc(&a_ptr, a_size));
  cuda_check(cudaMalloc(&b_ptr, b_size));
  cuda_check(cudaMalloc(&c_cuda_ptr, c_size));
  cuda_check(cudaMalloc(&c_cutlass_ptr, c_size));

  std::random_device rd{};
  fill_random_data(a_ptr, M * K, rd());
  fill_random_data(b_ptr, N * K, rd());

  std::vector<half> a_cpu(static_cast<size_t>(M * K), 0);
  std::vector<half> b_cpu(static_cast<size_t>(N * K), 0);
  std::vector<half> c_cpu(static_cast<size_t>(M * N), 0);
  cuda_check(cudaMemcpy(a_cpu.data(), a_ptr, a_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(b_cpu.data(), b_ptr, b_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cpu_gemm(M, N, K, a_cpu.data(), b_cpu.data(), c_cpu.data());
  cuda_check(launch_gemm_ampere_tcore_smem(a_ptr, b_ptr, M, N, K, c_cuda_ptr));
  CutlassGemmAmpereTcore(a_ptr, b_ptr, M, N, K, c_cutlass_ptr);

  std::vector<half> cuda_data(M * N);
  std::vector<half> cutlass_data(M * N);
  cuda_check(cudaMemcpy(cuda_data.data(), c_cuda_ptr, M * N * 2, cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(cutlass_data.data(), c_cutlass_ptr, M * N * 2, cudaMemcpyDeviceToHost));
  // std::cout << "Comparing cutlass with cpu" << "\n";
  // valid(c_cpu.data(), cutlass_data.data(), M, N);
  // std::cout << "Comparing cutlass with cuda" << "\n";
  // valid(cutlass_data.data(), cuda_data.data(), M, N);
  std::cout << "Comparing cuda with cpu" << "\n";
  valid(c_cpu.data(), cuda_data.data(), M, N);

  std::cout << "Run passed" << "\n";

  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
  cuda_check(cudaFree(c_cuda_ptr));
  cuda_check(cudaFree(c_cutlass_ptr));
}
