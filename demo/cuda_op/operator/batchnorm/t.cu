#include <cuda_runtime.h>
#include <cmath>

// 计算每个通道的均值和方差（使用共享内存归约）
__global__ void compute_mean_kernel(const float* input, float* mean, int C, int N, int H, int W) {
  extern __shared__ float shared[];
  int c            = blockIdx.x;
  int tid          = threadIdx.x;
  int num_elements = N * H * W;

  // 每个线程计算部分和
  float sum = 0.0f;
  for (int idx = tid; idx < num_elements; idx += blockDim.x) {
    int n      = idx / (H * W);
    int hw     = idx % (H * W);
    int h      = hw / W;
    int w      = hw % W;
    int index  = ((n * C + c) * H + h) * W + w;
    sum       += input[index];
  }

  // 存储到共享内存
  shared[tid] = sum;
  __syncthreads();

  // 归约求和
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  // 写入全局内存
  if (tid == 0) {
    mean[c] = shared[0] / num_elements;
  }
}

__global__ void
compute_var_kernel(const float* input, const float* mean, float* var, int C, int N, int H, int W) {
  extern __shared__ float shared[];
  int c            = blockIdx.x;
  int tid          = threadIdx.x;
  int num_elements = N * H * W;
  float mean_val   = mean[c];

  // 每个线程计算部分平方和
  float sum_sq = 0.0f;
  for (int idx = tid; idx < num_elements; idx += blockDim.x) {
    int n       = idx / (H * W);
    int hw      = idx % (H * W);
    int h       = hw / W;
    int w       = hw % W;
    int index   = ((n * C + c) * H + h) * W + w;
    float diff  = input[index] - mean_val;
    sum_sq     += diff * diff;
  }

  // 存储到共享内存
  shared[tid] = sum_sq;
  __syncthreads();

  // 归约求和
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  // 写入全局内存
  if (tid == 0) {
    var[c] = shared[0] / num_elements;
  }
}

// 更新移动平均值
__global__ void update_moving_average_kernel(
  float* running_mean,
  float* running_var,
  const float* batch_mean,
  const float* batch_var,
  float momentum,
  int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * batch_mean[c];
    running_var[c]  = momentum * running_var[c] + (1.0f - momentum) * batch_var[c];
  }
}

// 应用BatchNorm变换
__global__ void batchnorm_forward_kernel(
  const float* input,
  float* output,
  const float* gamma,
  const float* beta,
  const float* mean,
  const float* var,
  float epsilon,
  int C,
  int N,
  int H,
  int W) {
  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = N * C * H * W;

  if (idx < total_elements) {
    int n = idx / (C * H * W);
    int c = (idx / (H * W)) % C;
    int h = (idx / W) % H;
    int w = idx % W;

    float inv_std  = rsqrtf(var[c] + epsilon);
    float norm_val = (input[((n * C + c) * H + h) * W + w] - mean[c]) * inv_std;
    output[((n * C + c) * H + h) * W + w] = gamma[c] * norm_val + beta[c];
  }
}

// BatchNorm前向传播主函数
void batchnorm_forward(
  const float* input,
  float* output,
  const float* gamma,
  const float* beta,
  float* running_mean,
  float* running_var,
  int N,
  int C,
  int H,
  int W,
  float epsilon,
  float momentum,
  bool training) {
  // 分配临时内存
  float *d_batch_mean, *d_batch_var;
  cudaMalloc(&d_batch_mean, C * sizeof(float));
  cudaMalloc(&d_batch_var, C * sizeof(float));

  const int threads_per_block = 256;

  if (training) {
    // 计算当前批次的均值和方差
    dim3 grid(C);
    dim3 block(threads_per_block);
    int shared_mem = threads_per_block * sizeof(float);

    compute_mean_kernel<<<grid, block, shared_mem>>>(input, d_batch_mean, C, N, H, W);
    compute_var_kernel<<<grid, block, shared_mem>>>(input, d_batch_mean, d_batch_var, C, N, H, W);

    // 更新移动平均值
    dim3 update_grid((C + 255) / 256);
    dim3 update_block(256);
    update_moving_average_kernel<<<update_grid, update_block>>>(
      running_mean,
      running_var,
      d_batch_mean,
      d_batch_var,
      momentum,
      C);

    // 使用当前批次的统计量进行归一化
    int total_elements = N * C * H * W;
    dim3 norm_grid((total_elements + threads_per_block - 1) / threads_per_block);
    batchnorm_forward_kernel<<<norm_grid, threads_per_block>>>(
      input,
      output,
      gamma,
      beta,
      d_batch_mean,
      d_batch_var,
      epsilon,
      C,
      N,
      H,
      W);
  } else {
    // 推理模式：使用移动平均值
    int total_elements = N * C * H * W;
    dim3 norm_grid((total_elements + threads_per_block - 1) / threads_per_block);
    batchnorm_forward_kernel<<<norm_grid, threads_per_block>>>(
      input,
      output,
      gamma,
      beta,
      running_mean,
      running_var,
      epsilon,
      C,
      N,
      H,
      W);
  }

  // 清理临时内存
  cudaFree(d_batch_mean);
  cudaFree(d_batch_var);
}
