#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float* data;
	int*   shape;
	int    ndim;
	int    size;
	int    device; // 0->CPU, 1->CUDA
} ndarray_t;

ndarray_t* ndarray_create(int* shape, int ndim);
void ndarray_print(ndarray_t* arr);
void ndarray_free(ndarray_t* arr);

// Note that only arr->data is moved to cuda
void ndarray_to_cuda(ndarray_t* arr);
void ndarray_from_cuda(ndarray_t* arr);

// Extract last error from cuda (if there is one)
// defined in cuda_utils.cu
void cuda_get_err();

// Create some memory on device
// defined in cuda_utils.cu
float* cuda_create(size_t n);

// Sync (must be done before interfacing with host)
// defined in cuda_utils.cu
void cuda_sync();

// Allocates and copies memory to device
// defined in cuda_utils.cu
float* cuda_write(const float* h_data, size_t n);

// Given a pointer on the device,
// return a pointer to the host
// defined in cuda_utils.cu
float* cuda_read(float* d_data, size_t n);

// Given a cuda pointer, free that memory
// defined in cuda_utils.cu
void cuda_free(float* d_data);

// Add two vectors
// defined in cuda_ops.cu
ndarray_t* cuda_add(const ndarray_t *d_a, const ndarray_t *d_b);

// Multiply two vectors (element wise)
// defined in cuda_ops.cu
ndarray_t* cuda_elem_mul(const ndarray_t* d_a, const ndarray_t* d_b);

// Apply relu to every element of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_relu(const ndarray_t* d_in);

// Apply sigmoid to every element of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_sigmoid(const ndarray_t* d_in);

// Apply softmax along a given dimension of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_softmax(const ndarray_t* d_in, const int dim);

// Add a scalar to every element of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_scalar_add(const ndarray_t* d_in, const float scalar);

// Multiply a scalar to every element of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_scalar_mul(const ndarray_t* d_in, const float scalar);

// Divide every element of a vector by a scalar
// defined in cuda_ops.cu
ndarray_t* cuda_scalar_div(const ndarray_t* d_in, const float scalar);

// Raise every element of a vector to a certain power
// defined in cuda_ops.cu
ndarray_t* cuda_scalar_pow(const ndarray_t* d_in, const float power);

// Negate every element of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_neg(const ndarray_t* d_in);

// Apply elementwise absolute value to a vector
// defined in cuda_ops.cu
ndarray_t* cuda_abs(const ndarray_t* d_in);

// Sum all elements of a vector
// defined in cuda_ops.cu
ndarray_t* cuda_sum(const ndarray_t* d_in);

// Transpose a vector (must be on cuda)
// defined in cuda_ops.cu
ndarray_t* cuda_transpose(const ndarray_t* d_in);

// Perform matrix multiplication
// defined in cuda_ops.cu
ndarray_t* cuda_matmul(ndarray_t* d_a, ndarray_t* d_b);

#ifdef __cplusplus
}
#endif
