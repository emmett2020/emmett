#include <cstddef>
#include <cstring>

__global__ void relu(const float *x_buf, int N, float *y_buf) {
  std::size_t i = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (i < N) {
    y_buf[i] = max(0.F, x_buf[i]);
  }
}
