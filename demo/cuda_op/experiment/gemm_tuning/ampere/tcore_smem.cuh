#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>
#include <cooperative_groups.h>

using namespace nvcuda;                                                 // NOLINT

namespace {
#define OFFSET(row, col, ld) ((row) * (ld) + (col))                     // NOLINT
#define FLOAT4(pointer)      (reinterpret_cast<float4*>(&(pointer))[0]) // NOLINT

  template <unsigned BM, unsigned BN, unsigned BK, unsigned PAD>
  __global__ void gemm_tcore(half* A, half* B, unsigned M, unsigned N, unsigned K, half* C) {
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + 2 * BM * (BK + PAD); // Ping-pong needs two buffers.

    const int sa_tile = BM * (BK + PAD);
    const int sb_tile = BK * (BN + PAD);

    int sa_based_addr = __cvta_generic_to_shared(As); // NOLINT
    int sb_based_addr = __cvta_generic_to_shared(Bs); // NOLINT

    const unsigned bx  = blockIdx.x;
    const unsigned by  = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned wid = threadIdx.x / warpSize;

    const unsigned a_row_blk  = by * BM;
    const unsigned a_smem_row = tid / 4 * 2;
    const unsigned a_smem_col = tid % 4 * 8;
    const unsigned a_gmem_row = a_row_blk + a_smem_row;

    const unsigned b_col_blk  = bx * BN;
    const unsigned b_smem_row = tid / 32 * 4;
    const unsigned b_smem_col = (tid % 32) * 8;
    const unsigned b_gmem_col = b_col_blk + b_smem_col;

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

    const unsigned int T = K / BK;

    // Ping
    {
      const int t = 0; // first tile

      unsigned a_gemm_col    = t * BK + a_smem_col;
      unsigned a_gmem_offset = a_gmem_row * K + a_gemm_col;
      unsigned b_gmem_row    = t * BK + b_smem_row;

      int a_smem_ptr0 = sa_based_addr + OFFSET(a_smem_row + 0, a_smem_col, BK + PAD) * 2;
      int a_smem_ptr1 = sa_based_addr + OFFSET(a_smem_row + 1, a_smem_col, BK + PAD) * 2;

      int b_smem_ptr0 = sb_based_addr + OFFSET(b_smem_row + 0, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr1 = sb_based_addr + OFFSET(b_smem_row + 1, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr2 = sb_based_addr + OFFSET(b_smem_row + 2, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr3 = sb_based_addr + OFFSET(b_smem_row + 3, b_smem_col, BN + PAD) * 2;

      half* a_gmem_ptr0 = &A[a_gmem_offset];
      half* a_gmem_ptr1 = &A[a_gmem_offset + K];

      half* b_gmem_ptr0 = &B[(b_gmem_row + 0) * N + b_gmem_col];
      half* b_gmem_ptr1 = &B[(b_gmem_row + 1) * N + b_gmem_col];
      half* b_gmem_ptr2 = &B[(b_gmem_row + 2) * N + b_gmem_col];
      half* b_gmem_ptr3 = &B[(b_gmem_row + 3) * N + b_gmem_col];

      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(a_smem_ptr0), "l"(a_gmem_ptr0));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(a_smem_ptr1), "l"(a_gmem_ptr1));

      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr0), "l"(b_gmem_ptr0));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr1), "l"(b_gmem_ptr1));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr2), "l"(b_gmem_ptr2));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr3), "l"(b_gmem_ptr3));

      asm("cp.async.commit_group;\n" ::);
      asm("cp.async.wait_group 0;\n" ::);

      __syncthreads();
    }

    constexpr int KP = BK + PAD;
    constexpr int NP = BN + PAD;

    for (int t = 1; t < T; ++t) { // Starts from 1
      unsigned a_gmem_col    = t * BK + a_smem_col;
      unsigned a_gmem_offset = a_gmem_row * K + a_gmem_col;
      unsigned b_gmem_row    = t * BK + b_smem_row;

      int smem_idx_pong = (t & 1) ^ 1;       // Take
      int smem_idx_ping = ((t - 1) & 1) ^ 1; // Put

      int sa_addr_pi = sa_based_addr + smem_idx_ping * sa_tile * 2;
      int sb_addr_pi = sb_based_addr + smem_idx_ping * sb_tile * 2;

      int a_smem_ptr0 = sa_addr_pi + OFFSET(a_smem_row + 0, a_smem_col, BK + PAD) * 2;
      int a_smem_ptr1 = sa_addr_pi + OFFSET(a_smem_row + 1, a_smem_col, BK + PAD) * 2;

      int b_smem_ptr0 = sb_addr_pi + OFFSET(b_smem_row + 0, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr1 = sb_addr_pi + OFFSET(b_smem_row + 1, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr2 = sb_addr_pi + OFFSET(b_smem_row + 2, b_smem_col, BN + PAD) * 2;
      int b_smem_ptr3 = sb_addr_pi + OFFSET(b_smem_row + 3, b_smem_col, BN + PAD) * 2;

      half* a_gmem_ptr0 = &A[a_gmem_offset];
      half* a_gmem_ptr1 = &A[a_gmem_offset + K];

      half* b_gmem_ptr0 = &B[(b_gmem_row + 0) * N + b_gmem_col];
      half* b_gmem_ptr1 = &B[(b_gmem_row + 1) * N + b_gmem_col];
      half* b_gmem_ptr2 = &B[(b_gmem_row + 2) * N + b_gmem_col];
      half* b_gmem_ptr3 = &B[(b_gmem_row + 3) * N + b_gmem_col];

      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(a_smem_ptr0), "l"(a_gmem_ptr0));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(a_smem_ptr1), "l"(a_gmem_ptr1));

      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr0), "l"(b_gmem_ptr0));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr1), "l"(b_gmem_ptr1));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr2), "l"(b_gmem_ptr2));
      asm("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(b_smem_ptr3), "l"(b_gmem_ptr3));

      int sa_offset_po = smem_idx_pong * sa_tile;
      load_matrix_sync(a_frags[0][0], &As[sa_offset_po + (warp_load_m * 64 + 0) * KP + 0], KP);
      load_matrix_sync(a_frags[0][1], &As[sa_offset_po + (warp_load_m * 64 + 16) * KP + 0], KP);
      load_matrix_sync(a_frags[0][2], &As[sa_offset_po + (warp_load_m * 64 + 32) * KP + 0], KP);
      load_matrix_sync(a_frags[0][3], &As[sa_offset_po + (warp_load_m * 64 + 48) * KP + 0], KP);
      load_matrix_sync(a_frags[1][0], &As[sa_offset_po + (warp_load_m * 64 + 0) * KP + 16], KP);
      load_matrix_sync(a_frags[1][1], &As[sa_offset_po + (warp_load_m * 64 + 16) * KP + 16], KP);
      load_matrix_sync(a_frags[1][2], &As[sa_offset_po + (warp_load_m * 64 + 32) * KP + 16], KP);
      load_matrix_sync(a_frags[1][3], &As[sa_offset_po + (warp_load_m * 64 + 48) * KP + 16], KP);

      int sb_offset_po = smem_idx_pong * sb_tile;
      load_matrix_sync(b_frags[0][0], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 0], NP);
      load_matrix_sync(b_frags[0][1], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 16], NP);
      load_matrix_sync(b_frags[0][2], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 32], NP);
      load_matrix_sync(b_frags[0][3], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 48], NP);
      load_matrix_sync(b_frags[1][0], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 0], NP);
      load_matrix_sync(b_frags[1][1], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 16], NP);
      load_matrix_sync(b_frags[1][2], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 32], NP);
      load_matrix_sync(b_frags[1][3], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 48], NP);

#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          wmma::mma_sync(c_frags[i][j], a_frags[0][i], b_frags[0][j], c_frags[i][j]);
          wmma::mma_sync(c_frags[i][j], a_frags[1][i], b_frags[1][j], c_frags[i][j]);
        }
      }

      asm("cp.async.commit_group;\n" ::);
      asm("cp.async.wait_group 0;\n" ::);

      __syncthreads();
    }

    int smem_idx_pong = (T & 1) ^ 1; // Put
    int sa_offset_po  = smem_idx_pong * sa_tile;
    load_matrix_sync(a_frags[0][0], &As[sa_offset_po + (warp_load_m * 64 + 0) * KP + 0], KP);
    load_matrix_sync(a_frags[0][1], &As[sa_offset_po + (warp_load_m * 64 + 16) * KP + 0], KP);
    load_matrix_sync(a_frags[0][2], &As[sa_offset_po + (warp_load_m * 64 + 32) * KP + 0], KP);
    load_matrix_sync(a_frags[0][3], &As[sa_offset_po + (warp_load_m * 64 + 48) * KP + 0], KP);
    load_matrix_sync(a_frags[1][0], &As[sa_offset_po + (warp_load_m * 64 + 0) * KP + 16], KP);
    load_matrix_sync(a_frags[1][1], &As[sa_offset_po + (warp_load_m * 64 + 16) * KP + 16], KP);
    load_matrix_sync(a_frags[1][2], &As[sa_offset_po + (warp_load_m * 64 + 32) * KP + 16], KP);
    load_matrix_sync(a_frags[1][3], &As[sa_offset_po + (warp_load_m * 64 + 48) * KP + 16], KP);

    int sb_offset_po = smem_idx_pong * sb_tile;
    load_matrix_sync(b_frags[0][0], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 0], NP);
    load_matrix_sync(b_frags[0][1], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 16], NP);
    load_matrix_sync(b_frags[0][2], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 32], NP);
    load_matrix_sync(b_frags[0][3], &Bs[sb_offset_po + 0 * NP + warp_load_n * 64 + 48], NP);
    load_matrix_sync(b_frags[1][0], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 0], NP);
    load_matrix_sync(b_frags[1][1], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 16], NP);
    load_matrix_sync(b_frags[1][2], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 32], NP);
    load_matrix_sync(b_frags[1][3], &Bs[sb_offset_po + 16 * NP + warp_load_n * 64 + 48], NP);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wmma::mma_sync(c_frags[i][j], a_frags[0][i], b_frags[0][j], c_frags[i][j]);
        wmma::mma_sync(c_frags[i][j], a_frags[1][i], b_frags[1][j], c_frags[i][j]);
      }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned row_warp = a_row_blk + warp_load_m * 64 + i * 16;
        unsigned col_warp = b_col_blk + warp_load_n * 64 + j * 16;

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
    constexpr unsigned BK = 32;
    if (N % BN != 0 || M % BM != 0 || K % BK != 0) {
      throw std::runtime_error("not supported un-aligned case");
    }

    constexpr size_t num_threads = 256;

    const dim3 grid{N / BN, M / BM};
    constexpr unsigned PAD  = 0;
    constexpr unsigned smem = 2 * (BM * (BK + PAD) + BK * (BN + PAD)) * sizeof(half);
    gemm_tcore<BM, BN, BK, PAD><<<grid, num_threads, smem>>>(
      const_cast<half*>(A), // NOLINT
      const_cast<half*>(B), // NOLINT
      M,
      N,
      K,
      C);
    return cudaGetLastError();
  }

} // namespace

