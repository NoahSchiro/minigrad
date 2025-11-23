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

// Returns all combinations of indices excluding the axis dimension.
// Example: shape [2,3,4], axis=1 -> returns lists of form [i, *, k].
func AllIndexCombos(shape []int, axis int) [][]int {
    ndim := len(shape)
    if axis < 0 || axis >= ndim {
        panic("axis out of range")
    }

    // New shape with axis removed
    reduced := append([]int{}, shape[:axis]...)
    reduced = append(reduced, shape[axis+1:]...)

    total := 1
    for _, s := range reduced {
        total *= s
    }

    res := make([][]int, total)

    for i := 0; i < total; i++ {
        idx := make([]int, len(reduced))
        tmp := i
        for d := len(reduced) - 1; d >= 0; d-- {
            idx[d] = tmp % reduced[d]
            tmp /= reduced[d]
        }
        res[i] = idx
    }

    return res
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
