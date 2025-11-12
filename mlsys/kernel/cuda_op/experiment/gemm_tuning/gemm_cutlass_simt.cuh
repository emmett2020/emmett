#include <cutlass/gemm/device/gemm.h>

namespace {

  cudaError_t CutlassGemmAmpere(const float* A, const float* B, int M, int N, int K, float* C) {
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm =
      cutlass::gemm::device::Gemm< float, RowMajor, float, RowMajor, float, RowMajor >;
    CutlassGemm gemm_operator{};

    int lda = M;
    int ldb = K;
    int ldc = M;

    float alpha = 1.0F;
    float beta  = 0.0F;

    CutlassGemm::Arguments args{
      {M, N, K},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta}
    };

    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    return cudaSuccess;
  }
} // namespace
