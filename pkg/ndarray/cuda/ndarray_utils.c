#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ndarray.h"

ndarray_t* ndarray_create(int* shape, int ndim) {
	ndarray_t* res = (ndarray_t*)malloc(sizeof(ndarray_t));

	int size = 1;
	for (int i=0; i<ndim; i++) {
		size *= shape[i];
	}

	res->data = (float*)malloc(size * sizeof(float));
	res->shape =  (int*)malloc(ndim * sizeof(int));
	memcpy(res->shape, shape, ndim * sizeof(int));

	res->ndim = ndim;
	res->size = size;
	return res;
}


// Helper function: recursively print array
void ndarray_print_recursive(
	float* data,
	int* shape,
	int ndim,
	int indent,
	int* strides
) {
	
	// Print vector
    if (ndim == 1) {
        printf("%*s[", indent, "");
        for (int i = 0; i < shape[0]; i++) {
            printf("%.3f", data[i]);
            if (i < shape[0] - 1) printf(", ");
        }
        printf("]");

	// Print matrix
    } else if (ndim == 2) {
        printf("%*s[\n", indent, "");
        for (int i = 0; i < shape[0]; i++) {
            printf("%*s[", indent + 2, "");
            for (int j = 0; j < shape[1]; j++) {
                printf("%.3f", data[i * strides[0] + j * strides[1]]);
                if (j < shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < shape[0] - 1) printf(",\n");
            else printf("\n");
        }
        printf("%*s]", indent, "");

	// Print matrices (recursively)
    } else {
        printf("%*s[\n", indent, "");
        for (int i = 0; i < shape[0]; i++) {
            ndarray_print_recursive(data + i * strides[0], shape + 1, ndim - 1, indent + 2, strides + 1);
            if (i < shape[0] - 1) printf(",\n");
            else printf("\n");
        }
        printf("%*s]", indent, "");
    }
}

void ndarray_print(ndarray_t* arr) {
    if (!arr) return;

    // Compute strides
    int* strides = (int*) malloc(sizeof(int) * arr->ndim);
    strides[arr->ndim - 1] = 1;
    for (int i = arr->ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * arr->shape[i + 1];
    }

    ndarray_print_recursive(arr->data, arr->shape, arr->ndim, 0, strides);
    printf("\n");

    free(strides);
}



void ndarray_free(ndarray_t* arr) {
	free(arr->data);
	free(arr->shape);
	free(arr);
}
