#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>

using namespace nvcuda;                                                 // NOLINT

namespace {
#define OFFSET(row, col, ld) ((row) * (ld) + (col))                     // NOLINT
#define FLOAT4(pointer)      (reinterpret_cast<float4*>(&(pointer))[0]) // NOLINT

  template <unsigned BM, unsigned BN, unsigned BK>
  __global__ void gemm_tcore(half* A, half* B, unsigned M, unsigned N, unsigned K, half* C) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const unsigned bx  = blockIdx.x;
    const unsigned by  = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned wid = threadIdx.x / warpSize;

    const unsigned a_row_blk_start  = by * BM;
    const unsigned a_row_smem_start = tid / 4 * 2;
    const unsigned a_row_gmem_start = a_row_blk_start + a_row_smem_start;

    const unsigned b_col_blk_start = bx * BN;
    const unsigned b_col_smem      = (tid % 32) * 8;
    const unsigned b_col_gmem      = b_col_blk_start + b_col_smem;

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
      unsigned a_col_smem_start  = tid % 4 * 8;
      unsigned a_col_gmem_start  = t * BK + a_col_smem_start;
      unsigned a_col_gmem_offset = a_row_gmem_start * K + a_col_gmem_start;

      FLOAT4(As[a_row_smem_start][a_col_smem_start])     = FLOAT4(A[a_col_gmem_offset]);
      FLOAT4(As[a_row_smem_start + 1][a_col_smem_start]) = FLOAT4(A[a_col_gmem_offset + K]);

      // Load B to shared memory
      unsigned b_row_smem       = tid / 32 * 4;
      unsigned b_row_gmem_start = t * BK + b_row_smem;

      FLOAT4(Bs[b_row_smem + 0][b_col_smem]) = FLOAT4(B[(b_row_gmem_start + 0) * N + b_col_gmem]);
      FLOAT4(Bs[b_row_smem + 1][b_col_smem]) = FLOAT4(B[(b_row_gmem_start + 1) * N + b_col_gmem]);
      FLOAT4(Bs[b_row_smem + 2][b_col_smem]) = FLOAT4(B[(b_row_gmem_start + 2) * N + b_col_gmem]);
      FLOAT4(Bs[b_row_smem + 3][b_col_smem]) = FLOAT4(B[(b_row_gmem_start + 3) * N + b_col_gmem]);
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
    gemm_tcore<BM, BN, TK>
      <<<grid, num_threads>>>(const_cast<half*>(A), const_cast<half*>(B), M, N, K, C); // NOLINT
    return cudaGetLastError();
  }

} // namespace

