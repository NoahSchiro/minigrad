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
ndarray_t* cuda_add(const ndarray_t *d_a, const ndarray_t *d_b) {
	
	ndarray_t* d_c = ndarray_create(d_a->shape, d_a->ndim);
	ndarray_to_cuda(d_c);

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (d_c->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(
		d_a->data,
		d_b->data,
		d_c->data,
		d_c->size
	);
	cuda_get_err();

	return d_c;
}

__global__
void vector_elem_mul(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

ndarray_t* cuda_elem_mul(const ndarray_t* d_a, const ndarray_t* d_b) {
	ndarray_t* d_c = ndarray_create(d_a->shape, d_a->ndim); 
	ndarray_to_cuda(d_c);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_c->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_elem_mul<<<blocksPerGrid, threadsPerBlock>>>(
		d_a->data,
		d_b->data,
		d_c->data,
		d_c->size
	);
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
ndarray_t* cuda_relu(const ndarray_t* d_in) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    // Launch config
    int threadsPerBlock = 256;
    int blocksPerGrid = (d_in->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_relu<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, d_out->data, d_out->size
	);
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

ndarray_t* cuda_sigmoid(const ndarray_t* d_in) {
    ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_sigmoid<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, d_out->data, d_out->size
	);
    cuda_get_err();
    return d_out;
}

__global__
void vector_softmax(
    const float* in,
    float* out,
    int outer_size,
    int dim_size,
    int inner_size
) {
    int idx = blockIdx.x;
    if (idx >= outer_size * inner_size) return;

    int outer = idx / inner_size;
    int inner = idx % inner_size;

    const float* in_ptr  = in  + outer * dim_size * inner_size + inner;
    float*       out_ptr = out + outer * dim_size * inner_size + inner;

    // Find maximum for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; i++) {
        float v = in_ptr[i*inner_size];
        if (v > max_val) max_val = v;
    }

    // Exponentiate and sum
    float sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        float e = expf(in_ptr[i*inner_size] - max_val);
        out_ptr[i*inner_size] = e;
        sum += e;
    }

    // Normalize
    for (int i = 0; i < dim_size; i++) {
        out_ptr[i*inner_size] /= sum;
    }
}

ndarray_t* cuda_softmax(const ndarray_t* d_in, const int dim) {
    ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
    ndarray_to_cuda(d_out);

    // Compute sizes
    int outer_size = 1;
    int inner_size = 1;
    int dim_size   = d_in->shape[dim];

    for (int i = 0; i < dim; i++) {
        outer_size *= d_in->shape[i];
	}

    for (int i = dim + 1; i < d_in->ndim; i++) {
        inner_size *= d_in->shape[i];
	}

    int num_blocks = outer_size * inner_size;

    // Launch kernel
    vector_softmax<<<num_blocks, 1>>>(
        d_in->data,
        d_out->data,
        outer_size,
        dim_size,
        inner_size
    );
    return d_out;
}

__global__
void vector_scalar_add(const float* in, const float scalar, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i]+scalar;
    }
}

ndarray_t* cuda_scalar_add(const ndarray_t* d_in, const float scalar) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_add<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, scalar, d_out->data, d_out->size
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

ndarray_t* cuda_scalar_mul(const ndarray_t* d_in, const float scalar) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_mul<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, scalar, d_out->data, d_out->size
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

ndarray_t* cuda_scalar_div(const ndarray_t* d_in, const float scalar) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_div<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, scalar, d_out->data, d_out->size
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

ndarray_t* cuda_scalar_pow(const ndarray_t* d_in, const float power) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

    vector_scalar_pow<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, power, d_out->data, d_out->size
	);
    cuda_get_err();
    return d_out;
}

ndarray_t* cuda_neg(const ndarray_t* d_in) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

	// Reuse multiplication kernel
    vector_scalar_mul<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, -1, d_out->data, d_out->size
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

ndarray_t* cuda_abs(const ndarray_t* d_in) {
	ndarray_t* d_out = ndarray_create(d_in->shape, d_in->ndim);
	ndarray_to_cuda(d_out);

    int threadsPerBlock = 256;
    int blocksPerGrid = (d_out->size + threadsPerBlock - 1) / threadsPerBlock;

	// Reuse multiplication kernel
    vector_abs<<<blocksPerGrid, threadsPerBlock>>>(
		d_in->data, d_out->data, d_out->size
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

ndarray_t* cuda_sum(const ndarray_t* d_in) {
    int threads = 256;
	// For shared mem inside kernel
    int sharedMemSize = threads * sizeof(float);

	// Copy input
    float* d_prev = (float*)d_in->data;
    float* d_curr = nullptr;

    int curr_n = d_in->size;

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
        if (d_prev != d_in->data) {
            cuda_free(d_prev);
        }

        d_prev = d_curr;
        curr_n = blocks;
    }

	int shape[] = {1};
	ndarray_t* result = ndarray_create(shape, 1);
	ndarray_to_cuda(result);
	result->data = d_prev;

    return result;
}

// Warp size
#define TILE_DIM 32
// Block rows is chosen so that
// 32 * 8 = 256 -> optimal number of threads per block
// In effect, we will also copy 8 rows of a matrix
// at a time to and from the tile
#define BLOCK_ROWS 8

__global__
void vector_transpose(
    const float* in,
    float* out,
    int batch, // number of matrices in a batch
    int rows,  // matrix dims
    int cols   // matrix dims
) {

	// Shared mem visible to each thread in a block
	// Essentially we are transposing just a
	// small tile of the matrix and then writing it back
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

	// Each block along the z-axis of the
	// grid handles a different matrix. B indexes
	// this matrix
    int b = blockIdx.z;

	// Each thread in a block handles it's own
	// elem in the matrix, indexed by x and y for a 2d matrix
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

	// Starting index of matrix we are working on
    int batch_offset = b * rows * cols;

    // Load input into tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] =
                in[batch_offset + (y + j) * cols + x];
        }
    }

	// Threads need to finish writing before they start reading
    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store transposed tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            out[batch_offset + (y + j) * rows + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

ndarray_t* cuda_transpose(const ndarray_t* d_in) {
	if (!d_in->device) {
		printf("Input not on cuda device for transpose!");
		return nullptr;
	}
	int ndim = d_in->ndim;

	// Create new shape
	int* d_shape_out = (int*)malloc(ndim * sizeof(int));
	for (int i=0; i<ndim-2; i++) {
		d_shape_out[i] = d_in->shape[i];
	}
	// Transpose last two elements of shape
	d_shape_out[ndim-2] = d_in->shape[ndim-1];
	d_shape_out[ndim-1] = d_in->shape[ndim-2];

	// Create result ndarray
	ndarray_t* d_out = ndarray_create(d_shape_out, d_in->ndim);
	// Resultant shape is stored in ndarray_t so this is duplicate info
	free(d_shape_out);
	// Move to CUDA
	ndarray_to_cuda(d_out);

    // Compute batch size
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) {
        batch *= d_in->shape[i];
    }
    int rows = d_in->shape[ndim - 2];
    int cols = d_in->shape[ndim - 1];

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM,
        batch
    );

	vector_transpose<<<grid, block>>>(
        d_in->data,
        d_out->data,
        batch,
        rows,
        cols
    );
	cuda_get_err();

	return d_out;
}

__global__
void vector_mat_mul(
    const float* d_a, // Inputs
    const float* d_b,
    float* d_c,       // Outputs
    int batch_size,
    int m, int k, int n
) {
    int batch = blockIdx.z;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < m && col < n) {
        const float* a_ptr = d_a + batch * m * k;
        const float* b_ptr = d_b + batch * k * n;
        float*       c_ptr = d_c + batch * m * n;

        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a_ptr[row*k+i] * b_ptr[i*n+col];
        }

        c_ptr[row*n+col] = sum;
    }
}

ndarray_t* cuda_matmul(ndarray_t* d_a, ndarray_t* d_b) {
    
	int m = d_a->shape[d_a->ndim - 2];
    int k = d_a->shape[d_a->ndim - 1];
    int n = d_b->shape[d_b->ndim - 1];
    
	int batch_size = 1;
    for (int i = 0; i < d_a->ndim - 2; ++i) {
        batch_size *= d_a->shape[i];
    }

    // Construct output shape: [..., m, n]
    int* c_shape = (int*)malloc(sizeof(int) * d_a->ndim);
    for (int i = 0; i < d_a->ndim - 2; ++i) {
        c_shape[i] = d_a->shape[i];
    }
    c_shape[d_a->ndim - 2] = m;
    c_shape[d_a->ndim - 1] = n;

	// Create result ndarray
	ndarray_t* d_c = ndarray_create(c_shape, d_a->ndim);
	// Resultant shape is stored in ndarray_t so this is duplicate info
	free(c_shape);
	// Move to CUDA
	ndarray_to_cuda(d_c);

	// Launch configuration
    dim3 block(16, 16);
    dim3 grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y,
        batch_size
    );

    vector_mat_mul<<<grid, block>>>(
        d_a->data,
        d_b->data,
        d_c->data,
        batch_size,
        m, k, n
    );

	return d_c;
}
