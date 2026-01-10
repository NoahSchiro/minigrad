#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
