#include "cudnn_graph.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace {
  inline void cudnn_check(cudnnStatus_t err) {
    if (err != cudnnStatus_t::CUDNN_STATUS_SUCCESS) {
      throw std::runtime_error{cudnnGetErrorString(err)};
    }
  }

} // namespace

int main() noexcept(false) {
  cudnnHandle_t cudnn = nullptr;
  cudnn_check(cudnnCreate(&cudnn));

  // 配置Tensor Core
  cudnn_check(cudnnSetConvolutionMathType(cudnn, CUDNN_TENSOR_OP_MATH));

  // Tensor descriptor
  cudnnTensorDescriptor_t input_desc  = nullptr;
  cudnnTensorDescriptor_t output_desc = nullptr;
  cudnn_check(cudnnCreateTensorDescriptor(&input_desc));
  cudnn_check(cudnnCreateTensorDescriptor(&output_desc));

  // Parameters.
  int batch        = 1;
  int channels     = 64;
  int height       = 32;
  int width        = 32;
  int out_channels = 128;
  int kernel_size  = 3;
  int pad          = 1;
  int stride       = 1;

  // NCHW
  cudnn_check(cudnnSetTensor4dDescriptor(
    input_desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_HALF, // FP16使用Tensor Core
    batch,
    channels,
    height,
    width));

  // 卷积描述符
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnn_check(cudnnCreateConvolutionDescriptor(&conv_desc));
  cudnn_check(cudnnSetConvolution2dDescriptor(
    conv_desc,
    pad,
    pad,
    stride,
    stride,
    1,
    1,                 // 填充和步幅
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_HALF)); // FP16计算

  // 滤波器描述符
  cudnnFilterDescriptor_t filter_desc = nullptr;
  cudnn_check(cudnnCreateFilterDescriptor(&filter_desc));
  cudnn_check(cudnnSetFilter4dDescriptor(
    filter_desc,
    CUDNN_DATA_HALF,
    CUDNN_TENSOR_NCHW,
    out_channels,
    channels,
    kernel_size,
    kernel_size));

  // 查找最佳卷积算法 (启用Tensor Core)
  cudnnConvolutionFwdAlgoPerf_t perf_results;
  int returned_count = 0;
  cudnn_check(cudnnFindConvolutionForwardAlgorithm(
    cudnn,
    input_desc,
    filter_desc,
    conv_desc,
    output_desc,
    1, // 请求的算法数量
    &returned_count,
    &perf_results));

  // 输出描述符
  int out_n = 0;
  int out_c = 0;
  int out_h = 0;
  int out_w = 0;
  cudnn_check(cudnnGetConvolution2dForwardOutputDim(
    conv_desc,
    input_desc,
    filter_desc,
    &out_n,
    &out_c,
    &out_h,
    &out_w));
  cudnn_check(cudnnSetTensor4dDescriptor(
    output_desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_HALF,
    out_n,
    out_c,
    out_h,
    out_w));

  // 分配设备内存
  half *d_input, *d_output, *d_filter;
  size_t input_bytes  = batch * channels * height * width * sizeof(half);
  size_t filter_bytes = out_channels * channels * kernel_size * kernel_size * sizeof(half);
  size_t output_bytes = out_n * out_c * out_h * out_w * sizeof(half);

  cudaMalloc(&d_input, input_bytes);
  cudaMalloc(&d_output, output_bytes);
  cudaMalloc(&d_filter, filter_bytes);

  // 初始化数据 (实际应用中需填充真实数据)
  cudaMemset(d_input, 0, input_bytes);
  cudaMemset(d_filter, 0, filter_bytes);

  // 工作空间
  size_t workspace_size = perf_results.memory;
  void* d_workspace     = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&d_workspace, workspace_size);
  }

  // 执行卷积
  float alpha = 1.0F;
  float beta  = 0.0F;
  cudnn_check(cudnnConvolutionForward(
    cudnn,
    &alpha,
    input_desc,
    d_input,
    filter_desc,
    d_filter,
    conv_desc,
    perf_results.algo,
    d_workspace,
    workspace_size,
    &beta,
    output_desc,
    d_output));

  std::cout << "Convolution successful! Using algorithm: " << perf_results.algo << std::endl;

  // 清理资源
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_filter);
  if (d_workspace != nullptr) {
    cudaFree(d_workspace);
  }
  cudnnDestroy(cudnn);
  return 0;
}

