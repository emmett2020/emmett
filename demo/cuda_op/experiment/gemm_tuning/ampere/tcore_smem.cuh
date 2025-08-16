#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>

using namespace nvcuda; // NOLINT

namespace {
  template <unsigned BM, unsigned BN, unsigned BK>
  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, half* C) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const unsigned bx  = blockIdx.x;
    const unsigned by  = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned wid = threadIdx.x / warpSize;

    const unsigned a_row_blk_start  = by * BM;
    const unsigned a_row_smem_start = tid / 4 * 2;
    const unsigned a_row_gmem_start = a_row_blk_start + a_row_smem_start;

    const unsigned b_col_blk_start  = bx * BN;
    const unsigned b_col_smem_start = (tid % 32) * 8;
    const unsigned b_col_gmem_start = b_col_blk_start + b_col_smem_start;

    const unsigned warp_load_m = wid & 1;
    const unsigned warp_load_n = wid / 2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frags[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frags[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frags[4][4];

    for (auto& c_frag: c_frags) {
      for (auto& frag: c_frag) {
        wmma::fill_fragment(frag, 0.0F);
      }
    }

    unsigned int T = K / BK;
    for (int t = 0; t < T; ++t) {
      // Load A to shared memory
      unsigned a_col_smem_start = tid % 4 * 8;
      unsigned a_col_gmem_start = t * BK + a_col_smem_start;

      for (int i = 0; i < 8; ++i) {
        unsigned k_in_tile = a_col_smem_start + i;
        unsigned k         = a_col_gmem_start + i;

        As[a_row_smem_start][k_in_tile]     = A[a_row_gmem_start * K + k];
        As[a_row_smem_start + 1][k_in_tile] = A[(a_row_gmem_start + 1) * K + k];
      }

      // Load B to shared memory
      unsigned b_row_smem_start = tid / 32 * 4;
      unsigned b_row_gmem_start = t * BK + b_row_smem_start;
      for (int i = 0; i < 8; ++i) {
        unsigned b_col = b_col_smem_start + i;

        Bs[b_row_smem_start + 0][b_col] = B[(b_row_gmem_start + 0) * N + b_col_gmem_start + i];
        Bs[b_row_smem_start + 1][b_col] = B[(b_row_gmem_start + 1) * N + b_col_gmem_start + i];
        Bs[b_row_smem_start + 2][b_col] = B[(b_row_gmem_start + 2) * N + b_col_gmem_start + i];
        Bs[b_row_smem_start + 3][b_col] = B[(b_row_gmem_start + 3) * N + b_col_gmem_start + i];
      }
      __syncthreads();

      wmma::load_matrix_sync(a_frags[0][0], &As[warp_load_m * 64 + 0][0], BK);
      wmma::load_matrix_sync(a_frags[0][1], &As[warp_load_m * 64 + 16][0], BK);
      wmma::load_matrix_sync(a_frags[0][2], &As[warp_load_m * 64 + 32][0], BK);
      wmma::load_matrix_sync(a_frags[0][3], &As[warp_load_m * 64 + 48][0], BK);
      wmma::load_matrix_sync(a_frags[1][0], &As[warp_load_m * 64 + 0][16], BK);
      wmma::load_matrix_sync(a_frags[1][1], &As[warp_load_m * 64 + 16][16], BK);
      wmma::load_matrix_sync(a_frags[1][2], &As[warp_load_m * 64 + 32][16], BK);
      wmma::load_matrix_sync(a_frags[1][3], &As[warp_load_m * 64 + 48][16], BK);

      wmma::load_matrix_sync(b_frags[0][0], &Bs[0][warp_load_n * 64 + 0], BN);
      wmma::load_matrix_sync(b_frags[0][1], &Bs[0][warp_load_n * 64 + 16], BN);
      wmma::load_matrix_sync(b_frags[0][2], &Bs[0][warp_load_n * 64 + 32], BN);
      wmma::load_matrix_sync(b_frags[0][3], &Bs[0][warp_load_n * 64 + 48], BN);
      wmma::load_matrix_sync(b_frags[1][0], &Bs[16][warp_load_n * 64 + 0], BN);
      wmma::load_matrix_sync(b_frags[1][1], &Bs[16][warp_load_n * 64 + 16], BN);
      wmma::load_matrix_sync(b_frags[1][2], &Bs[16][warp_load_n * 64 + 32], BN);
      wmma::load_matrix_sync(b_frags[1][3], &Bs[16][warp_load_n * 64 + 48], BN);

#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          wmma::mma_sync(c_frags[i][j], a_frags[0][i], b_frags[0][j], c_frags[i][j]);
          wmma::mma_sync(c_frags[i][j], a_frags[1][i], b_frags[1][j], c_frags[i][j]);
        }
      }
      __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        unsigned row_warp = a_row_blk_start + warp_load_m * 64 + i * 16;
        unsigned col_warp = b_col_blk_start + warp_load_n * 64 + j * 16;

        half* C_ = C + row_warp * N + col_warp;
        wmma::store_matrix_sync(C_, c_frags[i][j], N, wmma::mem_row_major);
      }
    }
  }

  cudaError_t launch_gemm_ampere_tcore_smem(
    const half* A,
    const half* B,
    unsigned M,
    unsigned N,
    unsigned K,
    half* C) {
    constexpr unsigned BM = 128;
    constexpr unsigned BN = 256;
    constexpr unsigned TK = 32;
    if (N % BN != 0 || M % BM != 0 || K % TK != 0) {
      throw std::runtime_error("not supported un-aligned case");
    }

    constexpr size_t num_threads = 256;

    const dim3 grid{N / BN, M / BM};
    gemm_tcore<BM, BN, TK><<<grid, num_threads>>>(A, B, M, N, K, C);
    return cudaGetLastError();
  }

} // namespace

