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

extern "C"
float* cuda_create(size_t n) {
	float* d_data;
	cudaMalloc(&d_data, n*sizeof(float));
	cuda_get_err();
	return d_data;
}


extern "C"
float* cuda_write(const float* data, size_t n) {
	float* d_data;
	cudaMalloc(&d_data, n*sizeof(float));
	cuda_get_err();
	cudaMemcpy(d_data, data, n*sizeof(float), cudaMemcpyHostToDevice);
	cuda_get_err();
	return d_data;
}

extern "C"
float* cuda_read(float* d_data, size_t n) {
	float* h_data = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_data, d_data, n*sizeof(float), cudaMemcpyDeviceToHost);
	cuda_get_err();
	return h_data;
}

extern "C"
void cuda_free(float* d_data) {
	cudaFree(d_data);
	cuda_get_err();
}

// C wrapper code exposed by header
// Inputs must be on the GPU already and the output is placed on the GPU
extern "C"
float* cuda_vector_add(const float *d_a, const float *d_b, int n) {
	
	float *d_c = cuda_create(n); 

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
	cudaDeviceSynchronize();
	cuda_get_err();

	return d_c;
}
