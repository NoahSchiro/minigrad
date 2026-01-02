#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>

// Allocates and copies memory to device
float* cuda_create(const float* h_data, size_t n);

// Given a pointer on the host and on the device,
// copy data to host pointer.
void cuda_read(float* h_data, float* d_data, size_t n);

// Given a cuda pointer, free that memory
void cuda_free(float* d_data);

// C wrapper code to vector addition kernel
void cuda_vector_add(const float*, const float*, float*, int n);

#endif
