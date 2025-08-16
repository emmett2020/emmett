#include "cutlass/arch/mma.h"
#include "cutlass/half.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/cutlass.h>

namespace {


  cudaError_t CutlassGemmAmpereTcore(const half* A, const half* B, int M, int N, int K, half* C) {
    using RowMajor    = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,
      RowMajor,
      cutlass::half_t,
      RowMajor,
      cutlass::half_t,
      RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80>;

    CutlassGemm gemm_operator{};

    int lda = M;
    int ldb = K;
    int ldc = M;

    cutlass::half_t alpha(1.0F);
    cutlass::half_t beta(0.0F);

    // WARN: Mismatched, however, we don't care it's output now.
    CutlassGemm::Arguments args{
      {M, N, K},
      {reinterpret_cast<const cutlass::half_t*>(A), lda},
      {reinterpret_cast<const cutlass::half_t*>(B), ldb},
      {reinterpret_cast<cutlass::half_t*>(C), ldc},
      {reinterpret_cast<cutlass::half_t*>(C), ldc},
      {alpha, beta}
    };

    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    return cudaSuccess;
  }
} // namespace
