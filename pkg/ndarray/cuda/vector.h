#pragma once

#ifdef __cplusplus
extern "C" {
#endif


// Create some memory on device
float* cuda_create(size_t n);

// Allocates and copies memory to device
float* cuda_write(const float* h_data, size_t n);

// Given a pointer on the device,
// return a pointer to the host
float* cuda_read(float* d_data, size_t n);

// Given a cuda pointer, free that memory
void cuda_free(float* d_data);

// C wrapper code to vector addition kernel
float* cuda_vector_add(const float *d_a, const float *d_b, int n);

#ifdef __cplusplus
}
#endif
