#include <array>
#include <format>
#include <cstring>
#include <source_location>

namespace {
__global__ void VecAdd(const float *A, const float *B, float *C, int N) {
  std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

inline void cuda_check(cudaError_t err, std::source_location loc = std::source_location::current()) {
    if (err != cudaSuccess) {
      auto str = std::format(
        "[CUDA ERROR] at {}:{}:{}: {}",
        loc.file_name(),
        loc.line(),
        loc.function_name(),
        cudaGetErrorString(err));
      throw std::runtime_error(str);
    }
  }

} // namespace

int main() {
  constexpr std::size_t N = 1024;
  std::size_t byte_size = N * sizeof(float);

  // Allocate host memory.
  std::array<float, N> host_a{};
  std::array<float, N> host_b{};
  std::array<float, N> host_c{};

  // Initialize host buffer.
  std::memset(host_a.data(), 0, N);
  std::memset(host_b.data(), 0, N);

  // Allocate device memory
  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  cudaMalloc(&device_a, byte_size);
  cudaMalloc(&device_b, byte_size);
  cudaMalloc(&device_c, byte_size);

  // Copy buffers from host memory to device memory
  cudaMemcpy(device_a, host_a.data(), byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), byte_size, cudaMemcpyHostToDevice);

  // Invoke kernel
  constexpr int threads_per_block = 256;
  constexpr int blocks_per_grid =
      (N + threads_per_block - 1) / threads_per_block;
  VecAdd<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c,
                                                 N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(device_c, host_c.data(), byte_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}
