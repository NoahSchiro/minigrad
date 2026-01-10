#include <stdio.h>
#include <cuda_runtime.h>
#include "ndarray.h"

// Vector addition
__global__
void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// C wrapper code for vector_add
// Inputs must be on the GPU already and the output is placed on the GPU
extern "C"
float* cuda_vector_add(const float *d_a, const float *d_b, int n) {
	
	float *d_c = cuda_create(n); 

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
	cuda_get_err();

	return d_c;
}

// Vector unary apply of ReLU
__global__
void vector_relu(const float* in, float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

// C wrapper code for vector_add
// Inputs must be on the GPU already and the output is placed on the GPU
extern "C"
float* cuda_vector_relu(const float *d_in, int n) {
	float* d_out = cuda_create(n);

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_relu<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
	cuda_get_err();
	return d_out;

}

// Vector unary apply of Sigmoid
__global__
void vector_sigmoid(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C"
float* cuda_vector_sigmoid(const float* d_in, int n) {
    float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_sigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cuda_get_err();
    return d_out;
}
