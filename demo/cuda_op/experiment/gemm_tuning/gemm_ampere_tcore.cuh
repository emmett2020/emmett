#include <cstddef>
#include <iostream>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_base.cuh"

using namespace nvcuda; // NOLINT

namespace {
  inline constexpr std::size_t tile_size = 16;

  __global__ void
  gemm_tcore(const half* A, const half* B, unsigned M, unsigned N, unsigned K, float* C) {
    wmma::fragment<wmma::matrix_a, tile_size, tile_size, tile_size, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, tile_size, tile_size, tile_size, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, tile_size, tile_size, tile_size, float> c_frag;

    wmma::fill_fragment(c_frag, 0.F);

    wmma::load_matrix_sync(a_frag, A, tile_size);
    wmma::load_matrix_sync(b_frag, B, tile_size);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, tile_size, wmma::mem_row_major);
  }

  void launch_gemm_ampere_tcore(
    const float* A,
    const float* B,
    unsigned M,
    unsigned N,
    unsigned K,
    float* C) {
  }

} // namespace

