#include <stdio.h>
#include <cuda_runtime.h>


// Vector addition
__global__
void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C"
float* cuda_create(const float* data, size_t n) {
	float* d_data;
	cudaMalloc(&d_data, n);
	cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);
	return d_data;
}

extern "C"
void cuda_read(float* h_data, float* d_data, size_t n) {
    cudaMemcpy(h_data, d_data, n, cudaMemcpyDeviceToHost);
}

extern "C"
void cuda_free(float* d_data) {
	cudaFree(d_data);
}

// C wrapper code exposed by header
extern "C"
void cuda_vector_add(const float *h_a, const float *h_b, float *h_c, int n) {

    size_t size = n * sizeof(float);

	float *d_a = cuda_create(h_a, size);
	float *d_b = cuda_create(h_b, size);
	float *d_c = cuda_create(h_c, size);

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("GPU kernel elapsed time: %f ms\n", milliseconds);

    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}
