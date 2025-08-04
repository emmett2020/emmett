#include <iostream>
#include <cstdint>
#include <vector>

namespace {
  void cuda_check(cudaError_t err) {
    if (err != cudaError_t::cudaSuccess) {
      throw std::runtime_error{cudaGetErrorString(err)};
    }
  }

  constexpr int dim      = 64; // For convenience, dim is divisible by 32.
  constexpr int tile_dim = 32; // tile size

  /// Tiles need transpose.
  // Say we have (blockIdx.x, blockIdx.y) = (2, 1),
  // it will visit rows: [32, 63) and columns： [64, 95) of input
  // After transpose, it should visit rows: [64, 95) and columns: [32, 63) of output instead.
  // However threadIdx doesn't need to transpose, it's relative offset of tiles.
  // From thread perspective,
  // Say (threadIdx.x, threadIdx.y) = (2, 3) of block (0, 0)
  // It will visit (row, col) = (2, 3) of input (load),
  //               (row, col) = (2, 3) of shared memory (store),
  //               (row, col) = (3, 2) of shared memory (load),
  //               (row, col) = (2, 3) of output (store),
  // Say (threadIdx.x, threadIdx.y) = (2, 3) of block (2, 1)
  // It will visit (row, col) = (35, 66) of input (load),
  //               (row, col) = (35, 66) of shared memory (store), (Actually, it's only used for easier understand)
  //               (row, col) = (34, 67) of shared memory (load),
  //               (row, col) = (67, 34) of output (store),
  // You could think follows code have two phase. In the first, we only copy
  // the data from HBM to shared memory. You needn't consider transpose here.
  // In second phase, we'll copy shared memory data to output buffer, transpose
  // occurs in here.
  __global__ void transpose_no_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned col_input = blockIdx.x * tile_dim + threadIdx.x;
    unsigned row_input = blockIdx.y * tile_dim + threadIdx.y;

    // Load matrix to shared memory
    // Store shared memory is conflict-free here.
    tile[threadIdx.y][threadIdx.x] = in[row_input * dim + col_input];
    __syncthreads();

    // Store from shared memory to matrix
    // Load shared memory is bank conflicted.
    unsigned col_output                = blockIdx.y * tile_dim + threadIdx.x;
    unsigned row_output                = blockIdx.x * tile_dim + threadIdx.y;
    out[row_output * dim + col_output] = tile[threadIdx.x][threadIdx.y];
  }

  __global__ void transpose_with_swizzle(const int* in, int* out) {
    __shared__ int tile[tile_dim][tile_dim];

    unsigned col_input = blockIdx.x * tile_dim + threadIdx.x;
    unsigned row_input = blockIdx.y * tile_dim + threadIdx.y;

    // Load matrix to shared memory
    // Store shared memory is conflict-free here.
    tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[row_input * dim + col_input];

    __syncthreads();

    // Store from shared memory to matrix
    // Load shared memory is conflict-free.
    unsigned col_output                = blockIdx.y * tile_dim + threadIdx.x;
    unsigned row_output                = blockIdx.x * tile_dim + threadIdx.y;
    out[row_output * dim + col_output] = tile[threadIdx.x][threadIdx.x ^ threadIdx.y];
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
