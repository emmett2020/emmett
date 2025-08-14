#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_base.cuh"

using namespace nvcuda; // NOLINT

namespace {
  // Size per warp
  inline constexpr std::size_t tile_size = 16;

  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, half* C) {
    const unsigned wid       = threadIdx.x / warpSize;
    const unsigned num_warps = 2; // 0,1 handles same row different col

    unsigned row = blockIdx.y * num_warps * tile_size + wid / 2 * tile_size;
    unsigned col = blockIdx.x * num_warps * tile_size + wid % 2 * tile_size;

    if (row >= M || col >= N) {
      return;
    }

    wmma::fragment<wmma::accumulator, tile_size, tile_size, tile_size, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0F);

    unsigned int T = (K + tile_size - 1) / tile_size;
    for (int t = 0; t < T; ++t) {
      wmma::fragment<wmma::matrix_a, tile_size, tile_size, tile_size, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, tile_size, tile_size, tile_size, half, wmma::row_major> b_frag;

      const size_t A_col = t * tile_size;
      const half* A_     = A + row * K + A_col;
      wmma::load_matrix_sync(a_frag, A_, K);

      const size_t B_row = t * tile_size;
      const half* B_     = B + B_row * N + col;
      wmma::load_matrix_sync(b_frag, B_, N);

      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    half* C_ = C + row * N + col;
    wmma::store_matrix_sync(C_, c_frag, N, wmma::mem_row_major);
  }

  cudaError_t launch_gemm_ampere_tcore(
    const half* A,
    const half* B,
    unsigned M,
    unsigned N,
    unsigned K,
    half* C) {
    const dim3 block{128}; // 4 warps, 2 warps handle same row different col
    const unsigned tile_size_per_blk = 2 * tile_size;
    const dim3 grid{static_cast<unsigned int>((N + tile_size_per_blk - 1) / tile_size_per_blk),
                    static_cast<unsigned int>((M + tile_size_per_blk - 1) / tile_size_per_blk)};
    gemm_tcore<<<grid, block>>>(A, B, M, N, K, C);
    return cudaGetLastError();
  }

} // namespace

