#include "flash_attention/flash_attention.h"

namespace cuda_op {
  __inline__ __global__ void flash_attn(
    const float* Q,
    const float* K,
    const float* V,
    int N,
    int d,
    int Tc,
    int Tr,
    int Bc,
    int Br,
    float softmax_scale,
    float* l,
    float* m,
    float* O) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Offset into Q,K,V,O,l,m - different for each batch and head.
    int qkv_offset = (bx * gridDim.y * N * d) + (bx * N * d);
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi     = sram;
    float* Kj     = &sram[tile_size];
    float* Vj     = &sram[tile_size * 2];
    float* S      = &sram[tile_size * 3];

    for (int j = 0; j < Tc; ++j) {
      // Load Kj, Vj to SRAM
      for (int x = 0; x < d; x++) {
        Kj[tx * d + x] = K[qkv_offset + j * tile_size + tx * d + x];
        Vj[tx * d + x] = V[qkv_offset + j * tile_size + tx * d + x];
      }
      __syncthreads();
    }
  }


} // namespace cuda_op
