template <const int BM, const int BN, const int BK, const int TM, const int TN, const int TILE_N>
__global__ void block_swizzling(float* A, float* B, float* C, int M, int N, int K) {
  const dim3 swizzled_block_idx =
    get_swizzled_data_block_idx(gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, TILE_N);
  const int STRIDE = blockDim.x * blockDim.y;
  const int OFFSET = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];
  float sum[TM][TN] = {0.0};

  for (int idx = 0; idx < K / BK; ++idx) {
    for (int i = 0; i < BM * BK / STRIDE; ++i) {
      int offset          = i * STRIDE + OFFSET;
      int sa_row          = offset / BK;
      int sa_col          = offset % BK;
      s_a[sa_row][sa_col] = A[(swizzled_block_idx.y * BM + sa_row) * K + sa_col + idx * BK];
    }
    for (int i = 0; i < BN * BK / STRIDE; ++i) {
      int offset          = i * STRIDE + OFFSET;
      int sb_row          = offset / BN;
      int sb_col          = offset % BN;
      s_b[sb_row][sb_col] = B[(sb_row + idx * BK) * N + swizzled_block_idx.x * BN + sb_col];
    }
    __syncthreads();

    for (int i = 0; i < TM; ++i) {
      for (int j = 0; j < TN; ++j) {
        for (int k = 0; k < BK; ++k) {
          sum[i][j] += s_a[threadIdx.y * TM + i][k] * s_b[k][threadIdx.x * TN + j];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      C[(swizzled_block_idx.y * BM + threadIdx.y * TM + i)
        * N
        + swizzled_block_idx.x
        * BN
        + threadIdx.x
        * TN
        + j] = sum[i][j];
    }
  }
}
