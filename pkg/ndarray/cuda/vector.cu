#include <stdio.h>
#include <cuda_runtime.h>

__global__
void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// C wrapper code exposed by header
extern "C"
void vectorAddCuda(const float *a, const float *b, float *c, int n) {
	size_t size = n * sizeof(float);

	// Copy info over to the device
	float *cuda_a, *cuda_b, *cuda_c;
    cudaMalloc((void **)&cuda_a, size);
    cudaMalloc((void **)&cuda_b, size);
    cudaMalloc((void **)&cuda_c, size);

    cudaMemcpy(cuda_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, size, cudaMemcpyHostToDevice);

	// One thread per element of vector
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// Compute
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(cuda_a, cuda_b, cuda_c, n);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel launch error: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA sync error: %s\n", cudaGetErrorString(err));
	}

	// Get the answer out
	cudaMemcpy(c, cuda_c, size, cudaMemcpyDeviceToHost);

	// Free mem
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
}
