#include <stdio.h>
#include <cuda_runtime.h>
#include "ndarray.h"

void cuda_get_err() {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf(
			stderr,
			"CUDA error : (%d) %s.\n",
            (int)err, cudaGetErrorString(err)
		);
        exit(1);
    }
}

void cuda_sync() {
	cudaDeviceSynchronize();
}

float* cuda_create(size_t n) {
	float* d_data;
	cudaMalloc(&d_data, n*sizeof(float));
	cuda_get_err();
	return d_data;
}


float* cuda_write(const float* data, size_t n) {
	float* d_data;
	cudaMalloc(&d_data, n*sizeof(float));
	cuda_get_err();
	cudaMemcpy(d_data, data, n*sizeof(float), cudaMemcpyHostToDevice);
	cuda_get_err();
	return d_data;
}

float* cuda_read(float* d_data, size_t n) {
	float* h_data = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_data, d_data, n*sizeof(float), cudaMemcpyDeviceToHost);
	cuda_get_err();
	return h_data;
}

void cuda_free(float* d_data) {
	cudaFree(d_data);
	cuda_get_err();
}
