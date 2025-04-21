package ndarray

import (
	"strings"
	"fmt"
)

func checkShape(a NdArray, b NdArray) bool {

	if len(a.shape) != len(b.shape) {
		return false
	}

	for i := range a.shape {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}

	return true
}

func intArrayProduct(data []int) int {
	prod := 1
	for i := range data { prod *= data[i] }
	return prod
}

// Helper to compute the flat index from N-dimensional indices
func flatIndex(indices, shape []int) int {
    idx, stride := 0, 1
    for i := len(shape) - 1; i >= 0; i-- {
        idx += indices[i] * stride
        stride *= shape[i]
    }
    return idx
}

// Recursive pretty printer
func prettyPrintNd(data []float32, shape []int, indices []int, dim int) string {
    if dim == len(shape) {
        // At the innermost dimension, print the value
        idx := flatIndex(indices, shape)
        return fmt.Sprintf("%v", data[idx])
    }
    var b strings.Builder
    b.WriteString("[")
    for i := 0; i < shape[dim]; i++ {
        indices = append(indices, i)
        b.WriteString(prettyPrintNd(data, shape, indices, dim+1))
        indices = indices[:len(indices)-1]
        if i != shape[dim]-1 {
            b.WriteString(", ")
        }
    }
    b.WriteString("]")
    return b.String()
}


