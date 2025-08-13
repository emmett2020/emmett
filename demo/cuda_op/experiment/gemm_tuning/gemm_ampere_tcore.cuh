#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_base.cuh"

using namespace nvcuda; // NOLINT

namespace {
  inline constexpr std::size_t tile_size = 16;

  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, half* C) {
    const unsigned wid = threadIdx.x / warpSize;

    unsigned row = blockIdx.y * tile_size;
    unsigned col = blockIdx.x * tile_size;

    if (row >= M && col >= N) {
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

  void launch_gemm_ampere_tcore(
    const half* A,
    const half* B,
    unsigned M,
    unsigned N,
    unsigned K,
    half* C) {
    const dim3 block{32};
    const dim3 grid{static_cast<unsigned int>((N + tile_size - 1) / tile_size),
                    static_cast<unsigned int>((M + tile_size - 1) / tile_size)};
    gemm_tcore<<<grid, block>>>(A, B, M, N, K, C);
  }

} // namespace

