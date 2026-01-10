#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
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
float* cuda_add(const float *d_a, const float *d_b, int n) {
	
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
float* cuda_relu(const float *d_in, int n) {
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

float* cuda_sigmoid(const float* d_in, int n) {
    float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_sigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cuda_get_err();
    return d_out;
}

__global__
void vector_scalar_add(const float* in, const float scalar, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i]+scalar;
    }
}

float* cuda_scalar_add(const float *d_in, const float scalar, int n) {
	float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_add<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, scalar, d_out, n
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_scalar_mul(const float* in, const float scalar, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i]*scalar;
    }
}

float* cuda_scalar_mul(const float *d_in, const float scalar, int n) {
	float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_mul<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, scalar, d_out, n
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_scalar_div(const float* in, const float scalar, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i]/scalar;
    }
}

float* cuda_scalar_div(const float *d_in, const float scalar, int n) {
	float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_div<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, scalar, d_out, n
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_scalar_pow(const float* in, const float power, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = powf(in[i], power);
    }
}

float* cuda_scalar_pow(const float *d_in, const float power, int n) {
	float* d_out = cuda_create(n);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_pow<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, power, d_out, n
	);
    cuda_get_err();
    return d_out;
}

float* cuda_neg(const float *d_in, int n) {
	float* d_out = cuda_create(n);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// Reuse multiplication kernel
    vector_scalar_mul<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, -1, d_out, n
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_abs(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] >= 0 ? in[i] : in[i] * -1;
    }
}

float* cuda_abs(const float *d_in, int n) {
	float* d_out = cuda_create(n);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// Reuse multiplication kernel
    vector_abs<<<blocksPerGrid, threadsPerBlock>>>(
		d_in, d_out, n
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_sum_partial(const float* in, float* out, int n) {

	// Shared by all threads, faster than __global__ mem
    extern __shared__ float sdata[];

	// Index within block
    unsigned int tid = threadIdx.x;

	// Global index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Store data in shared memory
    sdata[tid] = (i < n) ? in[i] : 0.0f;
	// We just wrote, all threads need to sync before reading
    __syncthreads();

	// Takes half the block, sums that half, divides
	// the block in half and repeat
	// [a, b, c, d, e, f, g, h, i, j, k, l]
	// Iteration 1:
	// [a+g, b+h, c+i, d+j, e+k, f+l]
	// Iteration 2:
	// [a+g+d+j, b+h+e+k, c+i+f+l]
	// And so on...
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

	// Result for this block
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

float* cuda_sum(const float* d_in, int n) {
    int threads = 256;
	// For shared mem inside kernel
    int sharedMemSize = threads * sizeof(float);

	// Copy input
    float* d_prev = (float*)d_in;
    float* d_curr = nullptr;

    int curr_n = n;

	// Apply partial sums until we have one elem left
    while (curr_n > 1) {
        int blocks = (curr_n + threads - 1) / threads;

        d_curr = cuda_create(blocks);

        vector_sum_partial<<<blocks, threads, sharedMemSize>>>(
            d_prev, d_curr, curr_n
        );
        cuda_get_err();
        cuda_sync();

        // Free intermediate buffer (but not original input)
        if (d_prev != d_in) {
            cuda_free(d_prev);
        }

        d_prev = d_curr;
        curr_n = blocks;
    }

    return d_prev;
}
