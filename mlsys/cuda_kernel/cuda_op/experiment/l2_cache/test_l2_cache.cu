#include <cstddef>
#include <iostream>
#include <cstdint>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  void throw_if(bool cond, std::string_view msg) {
    if (cond) {
      throw std::runtime_error{msg.data()};
    }
  }

  // Per thread block per tile.
  // The x dimension of the grid traverses along the W dimension of the matrix.
  // input is col-major
  // Since input is col-major, the consecutive thread block alongside x
  // dimension will deal with far away data address. So it should has lower L2
  // Cache hit rate.
  __global__ void travel(float* input, unsigned H, unsigned W) {
    unsigned row = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned col = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < H && col < W) {
      unsigned idx = (col * H) + row;

      // Enhance L2 cache effect and make sure this piece of code is compiled.
      float value = 0.0F;
      for (int i = 0; i < 10; ++i) {
        value += input[idx];
      }
      if (value < 0) {
        input[idx] = value;
      }
    }
  }

  constexpr int block_size = 16; // TODO:

  void launch_travel(float* input, unsigned H, unsigned W) {
    // const int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((W + block_size - 1) / block_size, (H + block_size - 1) / block_size);
    travel<<<grid, block>>>(input, H, W);
    cuda_check(cudaGetLastError());
  }

  __global__ void travel_swizzle(float* input, unsigned H, unsigned W) {
    unsigned row = (blockIdx.x * blockDim.x) + threadIdx.y;
    unsigned col = (blockIdx.y * blockDim.y) + threadIdx.x;
    if (row < H && col < W) {
      unsigned idx = (col * H) + row;

      // Enhance L2 cache effect and make sure this piece of code is compiled.
      float value = 0.0F;
      for (int i = 0; i < 10; ++i) {
        value += input[idx];
      }
      if (value < 0) {
        input[idx] = value;
      }
    }
  }

  void launch_travel_swizzle(float* input, unsigned H, unsigned W) {
    // const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((H + block_size - 1) / block_size, (W + block_size - 1) / block_size);
    travel_swizzle<<<grid, block>>>(input, H, W);
    cuda_check(cudaGetLastError());
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

  const unsigned H  = 4'096;
  const unsigned W  = 128;
  const size_t size = static_cast<uint64_t>(H * W) * sizeof(float);
  std::cout << "H=" << H << ", W=" << W << "\n";

  float* a_ptr = nullptr;
  float* b_ptr = nullptr;
  cuda_check(cudaMalloc(&a_ptr, size));
  cuda_check(cudaMalloc(&b_ptr, size));

  cuda_check(cudaFuncSetAttribute(
    travel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    100 // 强制禁用Swizzle（仅Ampere+有效）
    ));

  launch_travel(a_ptr, H, W);
  launch_travel_swizzle(b_ptr, H, W);

  cuda_check(cudaGetLastError());
  cuda_check(cudaDeviceSynchronize());
  cuda_check(cudaFree(a_ptr));
  cuda_check(cudaFree(b_ptr));
}
