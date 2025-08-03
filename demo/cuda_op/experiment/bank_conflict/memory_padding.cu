#include <iostream>
#include <cstdint>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  __global__ void bank_conflict(float* data, int N) {
    __shared__ float smem[64];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    // Each bank is visited by two lane.
    // e.g.:
    // lane0,  lane16 -> bank0
    // lane1,  lane17 -> bank1
    // lane15, lane31 -> bank15
    unsigned bank_idx            = lane_id % 16;
    unsigned bank_element_offset = bank_idx;

    // e.g.:
    // bank0 is visited by lane0 and lane16,
    // lane0  visit the begginning of bank0
    // lane16 visit bank0 + 128 bytes
    unsigned sub_bank_idx            = lane_id / 16;
    unsigned sub_bank_element_offset = sub_bank_idx * 32;
    unsigned s_element_idx           = bank_element_offset + sub_bank_element_offset;

    smem[s_element_idx] = sinf(static_cast<float>(lane_id));
    __syncthreads();

    if (tid < N) {
      data[tid] = smem[s_element_idx];
    }
  }

  __global__ void bank_conflict2(float* data) {
    __shared__ float smem[32][32];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (int i = 0; i < 32; ++i) {
      smem[lane_id][i] = 0;
    }
    __syncthreads();

    for (int i = 0; i < 32; ++i) {
      data[tid] = smem[lane_id][i];
    }
  }

  __global__ void permute(float* data) {
    __shared__ float smem[32][32];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (auto& j: smem) {
      j[lane_id] = 0;
    }
    __syncthreads();

    for (auto& j: smem) {
      data[tid] = j[lane_id];
    }
  }

  __global__ void memory_padding(float* data) {
    __shared__ float smem[32][33];
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane_id = threadIdx.x % 32;

    for (int i = 0; i < 32; ++i) {
      smem[lane_id][i] = 0;
    }
    __syncthreads();

    for (int i = 0; i < 32; ++i) {
      data[tid] = smem[lane_id][i];
    }
  }

  constexpr int dim      = 32; // For convenience, dim is divisible by 32.
  constexpr int tile_dim = 32; // tile size

  __global__ void transpose_no_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned x = blockIdx.x * tile_dim + threadIdx.x;
    unsigned y = blockIdx.y * tile_dim + threadIdx.y;

    // Load matrix to shared memory
    tile[threadIdx.y][threadIdx.x] = in[y * dim + x];

    __syncthreads();

    // Store from shared memory to matrix
    unsigned x_out           = blockIdx.y * tile_dim + threadIdx.x;
    unsigned y_out           = blockIdx.x * tile_dim + threadIdx.y;
    out[y_out * dim + x_out] = tile[threadIdx.x][threadIdx.y];
  }

  __device__ __forceinline__ unsigned xor_swizzle_index(unsigned x, unsigned y) {
    // 使用XOR操作混合x和y的低位
    return (y * tile_dim + x) ^ ((y & 3) << 2);
  }

  __global__ void transpose_with_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned x = blockIdx.x * tile_dim + threadIdx.x;
    unsigned y = blockIdx.y * tile_dim + threadIdx.y;

    // 使用Swizzle索引写入共享内存
    unsigned swizzled_idx = xor_swizzle_index(threadIdx.x, threadIdx.y);
    unsigned swizzled_x   = swizzled_idx % tile_dim;
    unsigned swizzled_y   = swizzled_idx / tile_dim;

    tile[swizzled_y][swizzled_x] = in[y * dim + x];

    __syncthreads();

    int value = tile[swizzled_x][swizzled_y];

    // 转置写入输出
    unsigned x_out           = blockIdx.y * tile_dim + threadIdx.x;
    unsigned y_out           = blockIdx.x * tile_dim + threadIdx.y;
    out[y_out * dim + x_out] = value;
  }


} // namespace

auto main() noexcept(false) -> int {
  const int N            = 1 * 1'024 * 1'024;
  const std::size_t size = N * sizeof(float);
  float* buffer          = nullptr;
  cudaMalloc(&buffer, size);

  bank_conflict2<<<1, 32>>>(buffer);
  permute<<<1, 32>>>(buffer);
  memory_padding<<<1, 32>>>(buffer);

  const int matrix_n            = dim * dim;
  const std::size_t matrix_size = matrix_n * sizeof(int);
  int* matrix                   = nullptr;
  int* transposed_matrix        = nullptr;
  cudaMalloc(&matrix, matrix_size);
  cudaMalloc(&transposed_matrix, matrix_size);

  dim3 block_dim(tile_dim, tile_dim);
  dim3 grid_dim(dim / tile_dim, dim / tile_dim);
  transpose_no_swizzle<<<grid_dim, block_dim>>>(matrix, transposed_matrix);
  transpose_with_swizzle<<<grid_dim, block_dim>>>(matrix, transposed_matrix);

  cudaFree(buffer);
  cudaFree(matrix);
  cudaFree(transposed_matrix);
}
