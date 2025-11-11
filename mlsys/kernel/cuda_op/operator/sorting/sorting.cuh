#include <cuda_runtime.h>
#include <cfloat>  // Para FLT_MAX

#define THREADS 512

__device__ void swap(float &a, float &b, bool dir) {
    if ((a > b) == dir) {
        float tmp = a;
        a = b;
        b = tmp;
    }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k, int N) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

    unsigned int ixj = i ^ j;
    if (ixj > i && ixj < N) {
        if ((i & k) == 0) {
            swap(dev_values[i], dev_values[ixj], true);
        } else {
            swap(dev_values[i], dev_values[ixj], false);
        }
    }
}

__host__ __device__ int next_power_of_two(int n) {
    int pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    return pow2;
}

// data is a device pointer of size N
extern "C" void solve(float* data, int N) {
    int padded_N = next_power_of_two(N);

    // Crear buffer temporal en device (con padding)
    float* dev_padded;
    cudaMalloc(&dev_padded, padded_N * sizeof(float));

    // Copiar datos originales
    cudaMemcpy(dev_padded, data, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Rellenar con FLT_MAX
    float pad_value = FLT_MAX;
    for (int i = N; i < padded_N; ++i) {
        cudaMemcpy(dev_padded + i, &pad_value, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Configurar grid
    int blocks = (padded_N + THREADS - 1) / THREADS;
    dim3 grid(blocks);
    dim3 threads(THREADS);

    // Bitonic sort
    for (int k = 2; k <= padded_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<grid, threads>>>(dev_padded, j, k, padded_N);
            cudaDeviceSynchronize();
        }
    }

    // Copiar resultado ordenado a buffer original
    cudaMemcpy(data, dev_padded, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Liberar buffer
    cudaFree(dev_padded);
}

