
#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>

#include "scope.h"

namespace {
__global__ void relu(const float *x_buf, int N, float *y_buf) {
  std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (i < N) {
    y_buf[i] = max(static_cast<float>(0), x_buf[i]);
  }
}

template <class DType, std::size_t N>
void print_array(const std::array<DType, N> &arr) {
  for (auto data : arr) {
    std::cout << data << ", ";
  }
  std::cout << "\n";
}

} // namespace

int main() {
  constexpr int N = 1024;
  std::array<float, N> x_host{1};
  std::array<float, N> y_host{2};

  float *x_cuda = nullptr;
  float *y_cuda = nullptr;
  cudaMalloc(&x_cuda, N);
  cudaMalloc(&y_cuda, N);

  auto guard = scope_guard([&]() noexcept {
    cudaFree(x_cuda);
    cudaFree(y_cuda);
  });

  cudaMemcpy(x_cuda, x_host.data(), N, cudaMemcpyKind::cudaMemcpyHostToDevice);
  relu<<<1, 1>>>(x_cuda, N, y_cuda);

  cudaMemcpy(y_host.data(), y_cuda, N, cudaMemcpyKind::cudaMemcpyDeviceToHost);
  print_array(y_host);
}
