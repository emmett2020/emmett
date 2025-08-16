#include "cutlass/arch/mma.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/cutlass.h>
#include <stdexcept>

namespace {

  cudaError_t CutlassGemmAmpereTcore(const half* A, const half* B, int M, int N, int K, half* C) {
    using RowMajor    = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,                        // 输入矩阵A的数据类型
      RowMajor,
      cutlass::half_t,                        // 输入矩阵B的数据类型
      RowMajor,
      cutlass::half_t,                        // 输出矩阵C的数据类型
      RowMajor,
      cutlass::half_t,                        // 输出矩阵C的数据类型
      cutlass::arch::OpClassTensorOp,         // 使用Tensor Core操作
      cutlass::arch::Sm80,                    // 目标架构(Ampere)
      cutlass::gemm::GemmShape<128, 128, 32>, // 线程块处理的矩阵大小
      cutlass::gemm::GemmShape<64, 64, 32>,   // warp处理的矩阵大小
      cutlass::gemm::GemmShape<16, 8, 16>     // Tensor Core指令处理的矩阵大小
      >;


    CutlassGemm gemm_operator{};

    int lda = K;
    int ldb = N;
    int ldc = N;

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
      throw std::runtime_error{cutlass::cutlassGetStatusString(status)};
    }
    return cudaSuccess;
  }
} // namespace
