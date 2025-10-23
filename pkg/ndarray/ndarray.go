package ndarray

import "fmt"
import "math/rand/v2"

type NdArray struct {
	data []float32
	shape []int
	size int
	ndim int
}

// Create an empty array
func Empty() NdArray {
	return NdArray{
		data: []float32{0},
		shape: []int{1},
		size: 1,
		ndim: 1,
	}
}

// Given data and a shape, construct a new array
func New(data []float32, shape []int) NdArray {

	prod := intArrayProduct(shape)
	if prod != len(data) {
		panic("Error: Length of data and product of shape does not match")
	}

	return NdArray{
		data: data,
		shape: shape,
		size: prod,
		ndim: len(shape),
	}
}

// Given a number and a shape, create an array filled with that number
func NewFill(data float32, shape []int) NdArray {

	prod := intArrayProduct(shape)

	array := make([]float32, prod)
	for i := range array { array[i] = data }
	new_shape := make([]int, len(shape))
	copy(new_shape, shape)

	return NdArray{
		data: array,
		shape: new_shape,
		size: prod,
		ndim: len(shape),
	}
}

// Return a zeroed out array of specified shape
func Zero(shape []int) NdArray { return NewFill(0, shape) }

// Given a shape, init a random ndarray with floats in range [0,1]
func Rand(shape []int) NdArray {

	// Compute how much space we need
	prod := intArrayProduct(shape)
	data := make([]float32, prod)

	// Fill with random
	for i := range data {
		data[i] = rand.Float32()
	}

	return NdArray{
		data: data,
		shape: shape,
		size: prod,
		ndim: len(shape),
	}
}

func (a NdArray) Clone() NdArray {
	new_data := make([]float32, a.size)
	copy(new_data, a.data)
	new_shape := make([]int, a.ndim)
	copy(new_shape, a.shape)

	return NdArray{
		data: new_data,
		shape: new_shape,
		size: a.size,
		ndim: a.ndim,
	}
}

// Display
func (a NdArray) Print() string {
	s := prettyPrintNd(a.data, a.shape, []int{}, 0)
	return s
}

// Get a certain index within the nd array
func (a NdArray) Get(idxs []int) (float32, error) {
	if len(idxs) != a.ndim {
		return 0, fmt.Errorf("number of indices %d does not match array dimensions %d", len(idxs), len(a.shape))
	}

	index := 0
	stride := 1
	for i := a.ndim - 1; i >= 0; i-- {
		if idxs[i] < 0 || idxs[i] >= a.shape[i] {
			return 0, fmt.Errorf("index out of bounds: dimension %d, index %d out of range [0, %d]", i, idxs[i], a.shape[i])
		}
		index += idxs[i] * stride
		stride *= a.shape[i];
	}

	if index >= a.size {
		return 0, fmt.Errorf("calculated flat index %d is out of bounds for data slice of length %d", index, a.size)
	}
	return a.data[index], nil 
}

func (a *NdArray) Fill(value float32) {
	for i := 0; i < a.size; i++ {
		a.data[i] = value;
	}
}

// Getters
func (a NdArray) Shape() []int {
	shape := make([]int, a.ndim)
	copy(shape, a.shape)
	return shape
}
func (a NdArray) Size() int {
	return a.size
}
func (a NdArray) Ndim() int {
	return a.ndim
}
// End getters
