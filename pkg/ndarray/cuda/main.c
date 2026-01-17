// Some code to test that things work just in C before we try other stuff
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "ndarray.h"

void basic_ops() {
	int a_shape[] = {2,2};
	ndarray_t* a = ndarray_create(a_shape, 2);
	for (int i=0; i<a->size; i++) {
		if (i % 2 == 0) {
			a->data[i] = (float)i;
		} else {
			a->data[i] = (float)-i;
		}
	}
	printf("A:\n");
	ndarray_print(a);
	ndarray_to_cuda(a);
	
	/*
	int b_shape[] = {2,2};
	ndarray_t* b = ndarray_create(b_shape, 2);
	for (int i=0; i<b->size; i++) {
		b->data[i] = (float)i;
	}
	printf("B:\n");
	ndarray_print(b);
	ndarray_to_cuda(b);
	*/
	ndarray_t* result = cuda_softmax(a, 0);
	
	ndarray_free(a); //ndarray_free(b);

	ndarray_from_cuda(result);
	printf("Result:\n");
	ndarray_print(result);
	ndarray_free(result);
}

void transpose() {

	int shape[] = {3,2};
	ndarray_t* a = ndarray_create(shape, 2);
	for (int i=0; i<a->size; i++) {
		a->data[i] = (float)i;
	}
	ndarray_print(a);
	ndarray_to_cuda(a);

	ndarray_t* b = cuda_transpose(a);
	ndarray_free(a);
	ndarray_from_cuda(b);
	
	ndarray_print(b);
	ndarray_free(b);
}

void matmul() {

	// Create an a
	int a_shape[] = {3,2};
	ndarray_t* a = ndarray_create(a_shape, 2);
	for (int i=0; i<a->size; i++) {
		a->data[i] = (float)i;
	}
	printf("A:\n");
	ndarray_print(a);
	ndarray_to_cuda(a);

	// Create a b
	int b_shape[] = {2,3};
	ndarray_t* b = ndarray_create(b_shape, 2);
	for (int i=0; i<b->size; i++) {
		b->data[i] = (float)i;
	}
	printf("B:\n");
	ndarray_print(b);
	ndarray_to_cuda(b);

	// compute result on gpu
	ndarray_t* result = cuda_matmul(a, b);

	// No longer need a and b
	ndarray_free(a);
	ndarray_free(b);

	printf("result:\n");
	ndarray_from_cuda(result);
	ndarray_print(result);
	ndarray_free(result);
}

int main() {
	basic_ops();
	return 0;
}
