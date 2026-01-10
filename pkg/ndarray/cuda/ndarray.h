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

// C wrapper code to vector addition kernel
// defined in cuda_ops.cu
float* cuda_vector_add(const float *d_a, const float *d_b, int n);

// C wrapper code to vector unary apply of relu kernel
// defined in cuda_ops.cu
float* cuda_vector_relu(const float *d_in, int n);

// C wrapper code to vector unary apply of relu kernel
// defined in cuda_ops.cu
float* cuda_vector_sigmoid(const float *d_in, int n);

#ifdef __cplusplus
}
#endif
