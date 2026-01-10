// Some code to test that things work just in C before we try other stuff
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "ndarray.h"

int main() {
	// Init
	size_t n = 10;
	float* a = (float*)malloc(n * sizeof(float));
	//float* b = (float*)malloc(n * sizeof(float));
	for (size_t i=0; i<n; i++) {
		a[i] = -1.0;
		//b[i] = 1.0;
	}

	// Move a and b to device
	float* d_a = cuda_write(a, n);
	//float* d_b = cuda_write(b, n);

	// No longer need host a and b
	free(a); //free(b);

	float* d_c = cuda_vector_relu(d_a, n);

	cuda_sync();

	// Move output back to CPU
	float* h_c = cuda_read(d_c, n);

	// No longer need the device pointers
	cuda_free(d_a); /*cuda_free(d_b);*/ cuda_free(d_c);

	for (int i=0; i<n; i++) {
		printf("%f\n", h_c[i]);
	}

	// Free up the host c
	free(h_c);

	return 0;
}
