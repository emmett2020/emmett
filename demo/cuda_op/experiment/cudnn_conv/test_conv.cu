#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDNN(status)             \
  if (status != CUDNN_STATUS_SUCCESS) { \
    std::cerr                           \
      << "CuDNN failure: "              \
      << cudnnGetErrorString(status)    \
      << " at line "                    \
      << __LINE__                       \
      << std::endl;                     \
    exit(EXIT_FAILURE);                 \
  }

int main() {
  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  // 配置Tensor Core
  CHECK_CUDNN(cudnnSetConvolutionMathType(cudnn, CUDNN_TENSOR_OP_MATH));

  // 张量描述符
  cudnnTensorDescriptor_t input_desc, output_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  // 卷积参数
  int batch = 1, channels = 64, height = 32, width = 32;
  int out_channels = 128, kernel_size = 3, pad = 1, stride = 1;

  // 输入张量 (NCHW格式)
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
    input_desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_HALF, // FP16使用Tensor Core
    batch,
    channels,
    height,
    width));

  // 卷积描述符
  cudnnConvolutionDescriptor_t conv_desc;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
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
  cudnnFilterDescriptor_t filter_desc;
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(
    filter_desc,
    CUDNN_DATA_HALF,
    CUDNN_TENSOR_NCHW,
    out_channels,
    channels,
    kernel_size,
    kernel_size));

  // 查找最佳卷积算法 (启用Tensor Core)
  cudnnConvolutionFwdAlgoPerf_t perf_results;
  int returned_count;
  CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
    cudnn,
    input_desc,
    filter_desc,
    conv_desc,
    output_desc,
    1, // 请求的算法数量
    &returned_count,
    &perf_results));

  // 输出描述符
  int out_n, out_c, out_h, out_w;
  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
    conv_desc,
    input_desc,
    filter_desc,
    &out_n,
    &out_c,
    &out_h,
    &out_w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
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
  if (workspace_size > 0)
    cudaMalloc(&d_workspace, workspace_size);

  // 执行卷积
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionForward(
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
  if (d_workspace)
    cudaFree(d_workspace);
  cudnnDestroy(cudnn);
  return 0;
}

