#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda; // NOLINT

namespace {
  // Size per warp

  template <unsigned BM, unsigned BN, unsigned Tile>
  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, half* C) {
    __shared__ half As[BM][Tile];
    __shared__ half Bs[Tile][BN];

    const unsigned tid       = threadIdx.x;
    const unsigned wid       = threadIdx.x / warpSize;
    const unsigned num_warps = blockDim.x / warpSize;

    const unsigned row_in_block = tid / 2;
    const unsigned col_in_block = tid / 2;

    const unsigned row = blockIdx.y * BM + row_in_block;
    const unsigned col = blockIdx.x * BN + col_in_block;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frags[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frags[4][4];

    for (auto& c_frag: c_frags) {
      for (auto& frag: c_frag) {
        wmma::fill_fragment(frag, 0.0F);
      }
    }

    unsigned int T = K / Tile;
    for (int t = 0; t < T; ++t) {
      unsigned k_thread_start = tid % 2 * 8;
      for (int i = 0; i < 8; ++i) {
        unsigned k_in_tile = k_thread_start + i;
        unsigned k         = t * Tile + k_in_tile;

        As[row_in_block][k_in_tile] = A[row * K + k];
        Bs[k_in_tile][col_in_block] = B[k * N + col];
      }
      __syncthreads();

      wmma::load_matrix_sync(a_frags[wid], &As[wid * 16][0], Tile);

      for (int j = 0; j < 4; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &Bs[0][j * 16], BN);
        wmma::mma_sync(c_frags[wid][j], a_frags[wid], b_frag, c_frags[wid][j]);
      }
      __syncthreads();
    }

    for (int j = 0; j < 4; ++j) {
      unsigned row_warp = blockIdx.y * BM + wid * 16;
      unsigned col_warp = blockIdx.x * BN + j * 16;

      half* C_ = C + row_warp * N + col_warp;
      wmma::store_matrix_sync(C_, c_frags[wid][j], N, wmma::mem_row_major);
    }
  }

  cudaError_t launch_gemm_ampere_tcore_smem(
    const half* A,
    const half* B,
    unsigned M,
    unsigned N,
    unsigned K,
    half* C) {
    constexpr size_t tile_size   = 16;
    constexpr size_t num_threads = 128;
    constexpr size_t num_warps   = num_threads / 32;      // 4 warps
    constexpr size_t BM          = num_warps * tile_size; // 64 rows
    constexpr size_t BN          = num_warps * tile_size; // 64 cols

    const dim3 grid{static_cast<unsigned int>((N + BN - 1) / BN),
                    static_cast<unsigned int>((M + BM - 1) / BM)};
    gemm_tcore<BM, BN, tile_size><<<grid, num_threads>>>(A, B, M, N, K, C);
    return cudaGetLastError();
  }

} // namespace

