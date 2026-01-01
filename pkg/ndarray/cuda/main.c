// Some code to test that things work just in C before we try other stuff
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"

int main() {

	size_t n = 100;

	float* a = (float*)malloc(n * sizeof(float));
	float* b = (float*)malloc(n * sizeof(float));
	float* c = (float*)malloc(n * sizeof(float));

	for (size_t i=0; i<n; i++) {
		a[i] = 1.0;
		b[i] = 1.0;
	}

	vectorAddCuda(a, b, c, (int)n);

	// for (size_t i=0; i<n; i++) {
	// 	printf("%f\n", c[i]);
	// }

	return 0;
}
