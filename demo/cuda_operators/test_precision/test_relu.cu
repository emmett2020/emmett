#include "relu/relu.cuh"

// int main() {
//   constexpr int N = 1024;
//   std::array<float, N> x_host{1};
//   std::array<float, N> y_host{2};
//
//   float *x_cuda = nullptr;
//   float *y_cuda = nullptr;
//   cudaMalloc(&x_cuda, N);
//   cudaMalloc(&y_cuda, N);
//
//   auto guard = scope_guard([&]() noexcept {
//     cudaFree(x_cuda);
//     cudaFree(y_cuda);
//   });
//
//   cudaMemcpy(x_cuda, x_host.data(), N,
//   cudaMemcpyKind::cudaMemcpyHostToDevice); relu<<<1, 1>>>(x_cuda, N, y_cuda);
//
//   cudaMemcpy(y_host.data(), y_cuda, N,
//   cudaMemcpyKind::cudaMemcpyDeviceToHost); print_array(y_host);
// }
