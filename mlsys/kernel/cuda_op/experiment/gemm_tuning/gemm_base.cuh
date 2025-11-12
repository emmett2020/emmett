#pragma once

#include <stdexcept>
#include <string_view>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

namespace {
  void throw_if(bool cond, std::string_view msg) {
    if (cond) {
      throw std::runtime_error{msg.data()};
    }
  }

  __global__ void fill_random_data_kernel(curandState* state, float* data, int n, uint64_t seed) {
    unsigned idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
      curand_init(seed, idx, 0, &state[idx]);
      data[idx] = curand_uniform(&state[idx]);
    }
  }

  __global__ void fill_random_data_kernel(curandState* state, half* data, int n, uint64_t seed) {
    unsigned idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) {
      curand_init(seed, idx, 0, &state[idx]);
      data[idx] = curand_uniform(&state[idx]);
    }
  }

  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  void fill_random_data(float* d_data, int num_elements, uint64_t seed) {
    curandState* d_state = nullptr;

    cuda_check(cudaMalloc(&d_state, num_elements * sizeof(curandState)));

    int block_size = 256;
    int grid_size  = (num_elements + block_size - 1) / block_size;
    fill_random_data_kernel<<<grid_size, block_size>>>(d_state, d_data, num_elements, seed);
  }

  void fill_random_data(half* d_data, int num_elements, uint64_t seed) {
    curandState* d_state = nullptr;

    cuda_check(cudaMalloc(&d_state, num_elements * sizeof(curandState)));

    int block_size = 256;
    int grid_size  = (num_elements + block_size - 1) / block_size;
    fill_random_data_kernel<<<grid_size, block_size>>>(d_state, d_data, num_elements, seed);
  }

  void print_host_buffer(const float* h_data, int H, int W, std::string_view title = "") {
    throw_if(H <= 0, "H must be greater than 0");
    throw_if(W <= 0, "W must be greater than 0");
    int num_elements = H * W;
    if (!title.empty()) {
      std::cout << title << "\n";
    }
    for (int h = 0; h < H; h++) {
      std::cout << "h=" << h << "\t";
      for (int w = 0; w < W - 1; ++w) {
        std::cout << h_data[(h * W) + w] << ", ";
      }
      std::cout << h_data[(h * W) + W - 1] << "\n";
    }
  }

  void print_device_buffer(const float* d_data, int H, int W, std::string_view title = "") {
    throw_if(H <= 0, "H must be greater than 0");
    throw_if(W <= 0, "W must be greater than 0");
    int num_elements = H * W;
    size_t byte_size = num_elements * sizeof(float);
    std::vector<float> h_data(num_elements);
    cuda_check(cudaMemcpy(h_data.data(), d_data, byte_size, cudaMemcpyDeviceToHost));
    print_host_buffer(h_data.data(), H, W, title);
  }

} // namespace
