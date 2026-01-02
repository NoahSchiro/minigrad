// Some code to test that things work just in C before we try other stuff
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "vector.h"

void vectorAdd(const float* a, const float* b, float* c, int n) {
	for (int i=0; i<n; i++) {
		c[i] = a[i] + b[i];
	}
}

int main() {

	size_t n = 100000000;
	int cuda = 1;

	float* a = (float*)malloc(n * sizeof(float));
	float* b = (float*)malloc(n * sizeof(float));
	float* c = (float*)malloc(n * sizeof(float));

	for (size_t i=0; i<n; i++) {
		a[i] = 1.0;
		b[i] = 1.0;
	}

	if (cuda) {
		cuda_vector_add(a, b, c, (int)n);
	} else {
		float starttime = (float)clock() / CLOCKS_PER_SEC;
		vectorAdd(a,b,c,(int)n);
		float endtime = (float)clock() / CLOCKS_PER_SEC;
		printf("elapsed: %f\n", endtime - starttime);
	}
	// for (size_t i=0; i<n; i++) {
	// 	printf("%f\n", c[i]);
	// }


	return 0;
}
