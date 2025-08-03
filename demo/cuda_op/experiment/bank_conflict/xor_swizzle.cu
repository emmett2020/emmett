#include <iostream>
#include <cstdint>
#include <vector>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  constexpr int dim      = 32; // For convenience, dim is divisible by 32.
  constexpr int tile_dim = 32; // tile size

  __global__ void transpose_no_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned x = blockIdx.x * tile_dim + threadIdx.x;
    unsigned y = blockIdx.y * tile_dim + threadIdx.y;

    // Load matrix to shared memory
    // Store shared memory is conflict-free here.
    tile[threadIdx.y][threadIdx.x] = in[y * dim + x];

    __syncthreads();

    // Store from shared memory to matrix
    // Load shared memory is bank conflicted.
    unsigned x_out           = blockIdx.y * tile_dim + threadIdx.x;
    unsigned y_out           = blockIdx.x * tile_dim + threadIdx.y;
    out[y_out * dim + x_out] = tile[threadIdx.x][threadIdx.y];
  }

  __global__ void transpose_with_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned x = blockIdx.x * tile_dim + threadIdx.x;
    unsigned y = blockIdx.y * tile_dim + threadIdx.y;

    // Load matrix to shared memory
    // Store shared memory is conflict-free here.
    tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[y * dim + x];

    __syncthreads();

    // Store from shared memory to matrix
    // Load shared memory is conflict-free.
    unsigned x_out           = blockIdx.y * tile_dim + threadIdx.x;
    unsigned y_out           = blockIdx.x * tile_dim + threadIdx.y;
    out[y_out * dim + x_out] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];
  }

  bool validate_transpose(const int* original, const int* transposed) {
    for (int y = 0; y < dim; y++) {
      for (int x = 0; x < dim; x++) {
        if (original[y * dim + x] != transposed[x * dim + y]) {
          std::cout << "Validation failed at (" << y << ", " << x << ")\n";
          std::cout
            << "Original: "
            << original[y * dim + x]
            << " Transposed: "
            << transposed[x * dim + y]
            << '\n';
          return false;
        }
      }
    }
    return true;
  }
} // namespace

auto main() noexcept(false) -> int {
  const int matrix_n                 = dim * dim;
  const std::size_t matrix_byte_size = matrix_n * sizeof(int);

  std::vector<int> h_input(matrix_n);

  // Initialize host matrix.
  for (int i = 0; i < matrix_n; i++) {
    h_input[i] = i;
  }

  // Allocate cuda matrix.
  int* matrix                       = nullptr;
  int* transposed_matrix_no_swizzle = nullptr;
  int* transposed_matrix_swizzle    = nullptr;
  cuda_check(cudaMalloc(&matrix, matrix_byte_size));
  cuda_check(cudaMalloc(&transposed_matrix_swizzle, matrix_byte_size));
  cuda_check(cudaMalloc(&transposed_matrix_no_swizzle, matrix_byte_size));

  // Copy host matrix to cuda matrix.
  cuda_check(cudaMemcpy(matrix, h_input.data(), matrix_byte_size, cudaMemcpyHostToDevice));


  dim3 block_dim(tile_dim, tile_dim);
  dim3 grid_dim(dim / tile_dim, dim / tile_dim);

  // No swizzle.
  transpose_no_swizzle<<<grid_dim, block_dim>>>(matrix, transposed_matrix_no_swizzle);
  cuda_check(cudaDeviceSynchronize());

  // Xor swizzle.
  transpose_with_swizzle<<<grid_dim, block_dim>>>(matrix, transposed_matrix_swizzle);
  cuda_check(cudaDeviceSynchronize());

  // Copy back output.
  std::vector<int> h_output_no_swizzle(matrix_n, 0);
  std::vector<int> h_output_with_swizzle(matrix_n, 0);
  cuda_check(cudaMemcpy(
    h_output_no_swizzle.data(),
    transposed_matrix_no_swizzle,
    matrix_byte_size,
    cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(
    h_output_with_swizzle.data(),
    transposed_matrix_swizzle,
    matrix_byte_size,
    cudaMemcpyDeviceToHost));

  // Validation.
  bool valid_no_swizzle   = validate_transpose(h_input.data(), h_output_no_swizzle.data());
  bool valid_with_swizzle = validate_transpose(h_input.data(), h_output_with_swizzle.data());
  std::cout << "No swizzle validation: " << (valid_no_swizzle ? "PASSED" : "FAILED") << '\n';
  std::cout << "With swizzle validation: " << (valid_with_swizzle ? "PASSED" : "FAILED") << '\n';

  // Recycle resources
  cudaFree(matrix);
  cudaFree(transposed_matrix_no_swizzle);
  cudaFree(transposed_matrix_swizzle);
}
