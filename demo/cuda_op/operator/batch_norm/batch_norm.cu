#include "batch_norm/batch_norm.h"

#include <torch/extension.h>
#include "common/utils.h"

namespace cuda_op {

  __global__ void compute_mean_kernel(const float* input, float* mean, int C, int N, int H, int W) {
    extern __shared__ float shared[];
    const int c            = blockIdx.x;
    const int tid          = threadIdx.x;
    const int num_elements = N * H * W;

    float sum = 0.0f;
    for (int idx = tid; idx < num_elements; idx += blockDim.x) {
      int n      = idx / (H * W);
      int hw     = idx % (H * W);
      int h      = hw / W;
      int w      = hw % W;
      int index  = ((n * C + c) * H + h) * W + w;
      sum       += input[index];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }

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

    float sum_sq = 0.0f;
    for (int idx = tid; idx < num_elements; idx += blockDim.x) {
      int n      = idx / (H * W);
      int hw     = idx % (H * W);
      int h      = hw / W;
      int w      = hw % W;
      int index  = ((n * C + c) * H + h) * W + w;
      float diff = input[index] - mean_val;
      sum_sq     = diff * diff;
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (tid < s) {
        shared[tid] += shared[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      var[c] = shared[0] / num_elements;
    }
  }

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

  __global__ void batch_norm_forward_kernel(
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

  cudaError_t batch_norm_forward(
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
    float *d_batch_mean, *d_batch_var;
    cudaMalloc(&d_batch_mean, C * sizeof(float));
    cudaMalloc(&d_batch_var, C * sizeof(float));
    const int threads_per_block = 256;
    if (training) {
      dim3 grid(C);
      dim3 block(threads_per_block);
      int shared_mem = threads_per_block * sizeof(float);

      compute_mean_kernel<<<grid, block, shared_mem>>>(input, d_batch_mean, C, N, H, W);
      compute_var_kernel<<<grid, block, shared_mem>>>(input, d_batch_mean, d_batch_var, C, N, H, W);

      // Update
      dim3 update_grid((C + 255) / 256);
      dim3 update_block(256);
      update_moving_average_kernel<<<update_grid, update_block>>>(
        running_mean,
        running_var,
        d_batch_mean,
        d_batch_var,
        momentum,
        C);

      int total_elements = N * C * H * W;
      dim3 norm_grid((total_elements + threads_per_block - 1) / threads_per_block);
      batch_norm_forward_kernel<<<norm_grid, threads_per_block>>>(
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
      int total_elements = N * C * H * W;
      dim3 norm_grid((total_elements + threads_per_block - 1) / threads_per_block);
      batch_norm_forward_kernel<<<norm_grid, threads_per_block>>>(
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

    cudaFree(d_batch_mean);
    cudaFree(d_batch_var);

    return cudaGetLastError();
  }

  torch::Tensor torch_batch_norm(
    const torch::Tensor& input,
    torch::Tensor& running_mean,
    torch::Tensor& running_var,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon,
    float momentum,
    bool training) {
    auto output = torch::zero(input);
    auto shape  = input.sizes();
    TORCH_CHECK(shape.size() == 4, "shapes of input must be 4");
    TORCH_CHECK(
      input.is_cuda()
        && running_mean.is_cuda()
        && running_mean.is_cuda()
        && gamma.is_cuda()
        && beta.is_cuda(),
      "Tensors must be on CUDA device");

    const float* input_ptr  = input.data_ptr<float>();
    const float* gamma_ptr  = gamma.data_ptr<float>();
    const float* beta_ptr   = beta.data_ptr<float>();
    float* running_mean_ptr = running_mean.data_ptr<float>();
    float* running_var_ptr  = running_var.data_ptr<float>();
    float* output_ptr       = output.data_ptr<float>();
    cuda_check(batch_norm_forward(
      input_ptr,
      output_ptr,
      gamma_ptr,
      beta_ptr,
      running_mean_ptr,
      running_var_ptr,
      shape[0],
      shape[1],
      shape[2],
      shape[3],
      epsilon,
      momentum,
      training));
    cuda_check(cudaDeviceSynchronize());
    return output;
  }

} // namespace cuda_op
