#include <cuda_runtime.h>
#include <iostream>

namespace {
  template <int N>
  struct RegisterSpiller {
    __device__ static float compute(float init) {
      float v = init * N;
      return v + RegisterSpiller<N - 1>::compute(v);
    }
  };

  template <>
  struct RegisterSpiller<0> {
    __device__ static float compute(float init) {
      return init;
    }
  };

  __global__ __launch_bounds__(512, 4) void RegisterSpill(float *d_out) {
    float init         = static_cast<float>(threadIdx.x) * 0.1F;
    float result       = RegisterSpiller<32>::compute(init);
    d_out[threadIdx.x] = result;
  }
} // namespace

int main() {
  RegisterSpill<<<4, 256>>>(nullptr);
  return 0;
}
