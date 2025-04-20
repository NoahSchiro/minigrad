package ndarray

import (
	"errors"
	"fmt"
	"strings"
)

type NdArray struct {
	data []float32
	shape []int
	size int
}

func New(data []float32, shape []int) (NdArray, error) {

	empty := NdArray{
		data: make([]float32, 0),
		shape: make([]int, 0),
		size: 0,
	}

	// Check if there is a mismatch between shape and data
	prod := 1
	for i := range shape {
		prod *= shape[i]
	}
	if prod != len(data) {
		return empty, errors.New("Error: Length of data and product of shape does not match")
	}

	return NdArray{
		data: data,
		shape: shape,
		size: len(data),
	}, nil
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


func (a NdArray) Print() {
	s := prettyPrintNd(a.data, a.shape, []int{}, 0)
	fmt.Print(s)
}

// Getters
func (a NdArray) Shape() []int {
	return a.shape
}
func (a NdArray) Size() int {
	return a.size
}
// End getters
