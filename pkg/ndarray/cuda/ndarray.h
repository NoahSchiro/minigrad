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
float* cuda_add(const float *d_a, const float *d_b, int n);

// Multiply two vectors (element wise)
// defined in cuda_ops.cu
float* cuda_elem_mul(const float *d_a, const float *d_b, int n);

// Apply relu to every element of a vector
// defined in cuda_ops.cu
float* cuda_relu(const float *d_in, int n);

// Apply sigmoid to every element of a vector
// defined in cuda_ops.cu
float* cuda_sigmoid(const float *d_in, int n);

// Add a scalar to every element of a vector
// defined in cuda_ops.cu
float* cuda_scalar_add(const float *d_in, const float scalar, int n);

// Multiply a scalar to every element of a vector
// defined in cuda_ops.cu
float* cuda_scalar_mul(const float *d_in, const float scalar, int n);

// Divide every element of a vector by a scalar
// defined in cuda_ops.cu
float* cuda_scalar_div(const float *d_in, const float scalar, int n);

// Raise every element of a vector to a certain power
// defined in cuda_ops.cu
float* cuda_scalar_pow(const float *d_in, const float power, int n);

// Negate every element of a vector
// defined in cuda_ops.cu
float* cuda_neg(const float *d_in, int n);

// Apply elementwise absolute value to a vector
// defined in cuda_ops.cu
float* cuda_abs(const float *d_in, int n);

// Sum all elements of a vector
// defined in cuda_ops.cu
float* cuda_sum(const float *d_in, int n);

// Transpose a vector (must be on cuda)
ndarray_t* cuda_transpose(const ndarray_t* d_in);

ndarray_t* cuda_matmul(ndarray_t* d_a, ndarray_t* d_b);

#ifdef __cplusplus
}
#endif
