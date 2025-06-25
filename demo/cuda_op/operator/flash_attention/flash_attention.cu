#include "flash_attention/flash_attention.h"

#include <cmath>
#include <torch/extension.h>

#include "common/utils.h"

/// REFERENCE: https://github.com/tspeterkim/flash-attention-minimal

namespace cuda_op {
  __inline__ __global__ void flash_attn_fwd(
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
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
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

      for (int i = 0; i < Tr; ++i) {
        // Load Qi to SRAM, l and m to registers
        for (int x = 0; x < d; ++x) {
          Qi[tx * d + x] = Q[qkv_offset + tile_size * j + tx * d + x];
        }

        float row_m_prev = m[lm_offset + (i * Br) + tx];
        float row_l_prev = l[lm_offset + (i * Br) + tx];

        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; ++y) {
          float sum = 0;
          for (int x = 0; x < d; ++x) {
            sum += Qi[tx * d + x] * Kj[y * d + x];
          }
          sum            *= softmax_scale;
          S[tx * Br + y]  = sum;
          row_m           = std::max(row_m, sum);
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        for (int y = 0; y < Bc; ++y) {
          S[tx * Bc + y]  = __expf(S[tx * Bc + y] - row_m);
          row_l          += S[tx * Bc + y];
        }

        // Compute new m and l
        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev)
                        + (__expf(row_m - row_m_new) * row_l);

        // Write O, l, m to HBM
        for (int x = 0; x < d; ++x) {
          float pv = 0; // Pij * Vj
          for (int y = 0; y < Bc; ++y) {
            pv += S[tx * Bc + y] * Vj[y * d + x];
          }
          // TODO: formula
          O[qkv_offset + (tile_size * i) + (tx * d) + x] =
            (1 / row_l_new)
            * ((row_l_prev
                * __expf(row_m_prev - row_m_new)
                * O[qkv_offset + (tile_size * i) + (tx * d) + x])
               + (__expf(row_m - row_m_new) * pv));
        }
        m[lm_offset + i * Br + tx] = row_m_new;
        l[lm_offset + i * Br + tx] = row_l_new;
      }
      __syncthreads();
    }
  }

  torch::Tensor
  torch_flash_attn(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    const int Bc = 32;
    const int Br = 32;
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);

    const int Tc              = ceil(static_cast<float>(N) / Bc);
    const int Tr              = ceil(static_cast<float>(N) / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);

    auto cuda = torch::Device(torch::kCUDA);
    l         = l.to(cuda);
    m         = m.to(cuda);

    // Calculate SRAM size needed per block.
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh); // batch_size * num_headers
    dim3 block_dim(Bc);   // Bc threads per block
    flash_attn_fwd<<<grid_dim, block_dim, sram_size>>>(
      Q.data_ptr<float>(),
      K.data_ptr<float>(),
      V.data_ptr<float>(),
      N,
      d,
      Tc,
      Tr,
      Bc,
      Br,
      softmax_scale,
      l.data_ptr<float>(),
      m.data_ptr<float>(),
      O.data_ptr<float>());
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    return O;
  }


} // namespace cuda_op
