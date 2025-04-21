package ndarray

import (
	"fmt"
	"math/rand/v2"
)

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

	return NdArray{
		data: array,
		shape: shape,
		size: prod,
		ndim: len(shape),
	}
}

// Given a shape, init a random ndarray with floats in range [0,1]
// Note that this cannot fail
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

func (a NdArray) Print() {
	s := prettyPrintNd(a.data, a.shape, []int{}, 0)
	fmt.Println(s)
}

// Getters
func (a NdArray) Shape() []int {
	return a.shape
}
func (a NdArray) Size() int {
	return a.size
}
func (a NdArray) Ndim() int {
	return a.ndim
}
// End getters
