#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

using namespace nvcuda; // NOLINT

namespace {
  __global__ void tcore_kernel(half* data, int M, int N) {
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    load_matrix_sync(b_frag, data, N);
    if (threadIdx.x == 0) {
      printf("elmenets=%d\n", b_frag.num_elements);
      for (int n = 0; n < b_frag.num_elements; ++n) {
        printf("%f\n", float(b_frag.x[n]));
      }
    }
  }


} // namespace

int main() {
  int M                    = 128;
  int N                    = 128;
  const int E              = M * N;
  const unsigned byte_size = E * sizeof(half);
  half* ptr                = nullptr;

  cudaMalloc(&ptr, byte_size);

  std::vector<half> host(E);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      host[m * N + n] = float(m);
      // printf("m=%d, n=%d, data=%f\n", m, n, float(host[m * N + n]));
    }
  }

  cudaMemcpy(ptr, host.data(), byte_size, cudaMemcpyHostToDevice);
  const dim3 grid_dim = 1;
  const dim3 blk_dim  = 32;

  tcore_kernel<<<grid_dim, blk_dim>>>(ptr, M, N);
  cudaDeviceSynchronize();
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

  return 0;
}
