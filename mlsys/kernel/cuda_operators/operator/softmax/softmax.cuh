#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <format>
#include <source_location>

namespace {
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

  // Get maximum of per thread block.
  // https://forums.developer.nvidia.com/t/finding-max-in-array/27666/14
  template <class T>
  __global__ void reduce_max_kernel(const T* input, std::size_t n, T* output) {
    extern __shared__ unsigned char char_shared_max[];
    T* shared_max = (T*) char_shared_max;
    const int tid = threadIdx.x;

    auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto local_max  = global_idx < n ? input[global_idx] : std::numeric_limits<T>::lowest();
    if (global_idx + blockDim.x < n) {
      T value = input[global_idx + blockDim.x];
      if (value > local_max) {
        local_max = value;
      }
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Get block max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        if (shared_max[tid + stride] > shared_max[tid]) {
          shared_max[tid] = shared_max[tid + stride];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      output[blockIdx.x] = shared_max[0];
    }
  }

  template <class T>
  void reduce_max(const T* input, std::size_t n, T* output) {
    const int block_size = 256;
    int grid_size        = (n + block_size - 1) / block_size;
    reduce_max_kernel<<<grid_size, block_size / 2, block_size * sizeof(T)>>>(input, n, output);
    int remaining = grid_size;
    while (remaining > 1) {
      grid_size = (remaining + block_size - 1) / block_size;
      reduce_max_kernel<<<grid_size, block_size / 2>>>(input, remaining, output);
      remaining = grid_size;
    }
  }

  template <class T>
  __global__ void exp_sum_kernel(const T* input, std::size_t n, T max, T* sum) {
    __shared__ extern unsigned char shared_max_temp[];
    auto* shared_sum = reinterpret_cast<T*>(shared_max_temp);
    //
    shared_sum[threadIdx.x] = 0;

    int origin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = origin_idx; idx < n; idx += gridDim.x * blockDim.x) {
      T exp_val                = expf(input[idx] - max);
      shared_sum[threadIdx.x] += exp_val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      sum[blockIdx.x] = 0;
      for (int i = 0; i < blockDim.x; ++i) {
        sum[blockIdx.x] += shared_sum[i];
      }
    }
  }

  template <class T>
  __global__ void result_kernel(const T* input, std::size_t n, T max, T sum, T* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
      T exp_val = expf(input[i] - max);
      output[i] = exp_val / sum;
    }
  }

  //
  template <class DType>
  void launch_softmax(const DType* input, std::size_t n, DType* output) {
    // Get maximum of thread block
    constexpr int threads_per_blk = 256;
    constexpr int num_blk         = 828;
    auto blk_maximums_cpu         = std::array<DType, num_blk >{};

    DType* blk_maximums_gpu = nullptr;
    cudaMalloc(&blk_maximums_gpu, num_blk * sizeof(DType));
    reduce_max(input, n, blk_maximums_gpu);
    cudaMemcpy(blk_maximums_cpu.data(),
               blk_maximums_gpu,
               num_blk * sizeof(DType),
               cudaMemcpyDeviceToHost);

    // Get global maximum
    DType max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < num_blk; ++i) {
      max = std::max(max, blk_maximums_cpu[i]);
    }

    // Calculate each exp
    auto acc       = std::array<DType, num_blk>{};
    DType* acc_gpu = nullptr;
    cudaMalloc(&acc_gpu, num_blk * sizeof(DType));
    constexpr std::size_t shared_byte_size = threads_per_blk * sizeof(DType);

    exp_sum_kernel<<<num_blk, threads_per_blk, shared_byte_size>>>(input, n, max, acc_gpu);
    cudaMemcpy(acc.data(), acc_gpu, num_blk * sizeof(DType), cudaMemcpyDeviceToHost);
    auto sum = DType{0};
    for (int i = 0; i < num_blk; ++i) {
      sum += acc[i];
    }
    result_kernel<<<num_blk, threads_per_blk>>>(input, n, max, sum, output);

    cudaDeviceSynchronize();
    auto last_error = cudaGetLastError();
    cuda_check(cudaGetLastError());
    float acc_f = static_cast<float>(acc[0]);
    printf("max=%f, acc=%f\n", float(max), acc_f);
  }
} // namespace


