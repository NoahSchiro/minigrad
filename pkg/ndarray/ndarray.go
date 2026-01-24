package ndarray

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#cgo LDFLAGS: -L${SRCDIR}/cuda -lcuda
#include <string.h>
#include "ndarray.h"
*/
import "C"

import (
	"unsafe"
	"fmt"
	"math/rand/v2"
)

type Device int

const (
	CPU Device = iota // Device = 0
	CUDA              // Device = 1
)

type NdArray struct {
	data []float32           // Implicitly, this is cpu only
	gpuData unsafe.Pointer
	shape []int
	size int
	ndim int
	device Device
}

// Create an empty array
func Empty() NdArray {
	return NdArray{
		data: []float32{0},
		shape: []int{1},
		size: 1,
		ndim: 1,
		device: CPU,
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
		device: CPU,
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
		device: CPU,
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
		device: CPU,
	}
}

// Given a shape, init a random ndarray with floats in range [0,1]
func Uniform(shape []int, lower, upper float32) NdArray {

	// Compute how much space we need
	prod := intArrayProduct(shape)
	data := make([]float32, prod)

	// Fill with uniform random number in range
	for i := range data {
		data[i] = lower + rand.Float32()*(upper-lower)
	}

	return NdArray{
		data: data,
		shape: shape,
		size: prod,
		ndim: len(shape),
		device: CPU,
	}
}

func (a NdArray) Clone() NdArray {
	
	new_shape := make([]int, a.ndim)
	copy(new_shape, a.shape)

	// Cuda
	if a.device == CUDA {

		cArr := (*C.ndarray_t)(a.gpuData);
		newcArr := unsafe.Pointer(C.ndarray_clone(cArr))

		return NdArray{
			data: nil,
			gpuData: newcArr,
			shape: new_shape,
			size: a.size,
			ndim: a.ndim,
			device: CUDA,
		}

	// CPU
	} else if a.device == CPU {
		new_data := make([]float32, a.size)
		copy(new_data, a.data)

		return NdArray{
			data: new_data,
			shape: new_shape,
			size: a.size,
			ndim: a.ndim,
			device: CPU,
		}
	} else {
		panic("Device not supported in ndarray.Clone!")
	}
}

// Move NdArray data to a particular device
// This assumes that the Go NdArray struct has the correct shape, ndim, etc
func (a *NdArray) To(device Device) {

	if a.device == CPU && device == CUDA {
        
		// Convert go int (64-bit) to C int (32-bit)
		cShape := make([]C.int, len(a.shape))
		for i, v := range a.shape {
			cShape[i] = C.int(v)
		}

		// Create C CPU ndarray_t
		cArr := C.ndarray_create_with_data(
			(*C.float)(unsafe.Pointer(&a.data[0])),
			(*C.int)(unsafe.Pointer(&cShape[0])),
			C.int(a.ndim),
		)

		// Move C CPU -> C CUDA
		C.ndarray_to_cuda(cArr)
		
		// Save pointer to C ndarray
		a.gpuData = unsafe.Pointer(cArr)
		a.device = CUDA
		
	} else if a.device == CUDA && device == CPU {
		cArr := (*C.ndarray_t)(a.gpuData)

		C.ndarray_from_cuda(cArr);

		// Convert C pointer to Go array pointer
		temp := (*[1 << 30]C.float)(unsafe.Pointer(cArr.data))[:a.size:a.size]

		// Allocate slice
		a.data = make([]float32, a.size)

		// Copy data
		for i, v := range temp {
			a.data[i] = float32(v)
		}

		// Free the c stuff
		C.ndarray_free(cArr);

		a.device = CPU
	}
}

// Display
func (a NdArray) Print() string {
	if a.device == CUDA {
		return "cannot print a NdArray on the GPU!"
	} else {
		return prettyPrintNd(a.data, a.shape, []int{}, 0)
	}
}

// Get a certain index within the nd array
func (a NdArray) Get(idxs []int) float32 {
	if len(idxs) != a.ndim {
		panic("number of indices does not match array dimensions")
	}

	index := 0
	stride := 1
	for i := a.ndim - 1; i >= 0; i-- {
		if idxs[i] < 0 || idxs[i] >= a.shape[i] {
			panic("index out of bounds")
		}
		index += idxs[i] * stride
		stride *= a.shape[i];
	}

	if index >= a.size {
		panic("calculated flat index is out of bounds for data slice")
	}
	return a.data[index]
}
func (a NdArray) GetLinear(idx int) (float32) {
	if idx >= a.size {
		panic(fmt.Sprintf("NdArray GetLinear: Index %d is greater than size %d", idx, a.size))
	}
	return a.data[idx]
}

func (a NdArray) Set(idxs []int, value float32) {
	if len(idxs) != a.ndim {
		fmt.Errorf("number of indices %d does not match array dimensions %d", len(idxs), len(a.shape))
		return
	}

	index := 0
	stride := 1
	for i := a.ndim - 1; i >= 0; i-- {
		if idxs[i] < 0 || idxs[i] >= a.shape[i] {
			fmt.Errorf("index out of bounds: dimension %d, index %d out of range [0, %d]", i, idxs[i], a.shape[i])
			return
		}
		index += idxs[i] * stride
		stride *= a.shape[i];
	}

	if index >= a.size {
		fmt.Errorf("calculated flat index %d is out of bounds for data slice of length %d", index, a.size)
		return
	}

	a.data[index] = value
}
func (a NdArray) SetLinear(idx int, value float32) {
	if idx >= a.size {
		panic(fmt.Sprintf("NdArray SetLinear: Index %d is greater than size %d", idx, a.size))
	}
	a.data[idx] = value
}

// Transpose
func (a NdArray) T() NdArray {
	if a.ndim < 2 {
		panic("NdArray.T: requires at least 2 dimensions")
	}

	// If NdArray is a matrix,
	// do a direct transpose
	if a.ndim == 2 {
		rows, cols := a.shape[0], a.shape[1]
		out := make([]float32, a.size)

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[j*rows+i] = a.data[i*cols+j]
			}
		}

		return NdArray{
			data:  out,
			shape: []int{cols, rows},
			size:  a.size,
			ndim:  2,
		}
	}

	// Ndim > 2 -> iterate over submatrices
	batch := 1
	for i := 0; i < a.ndim-2; i++ {
		batch *= a.shape[i]
	}
	rows := a.shape[a.ndim-2]
	cols := a.shape[a.ndim-1]

	out := make([]float32, a.size)
	submatSize := rows * cols

	for b := 0; b < batch; b++ {
		offset := b * submatSize
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[offset+j*rows+i] = a.data[offset+i*cols+j]
			}
		}
	}

	newShape := append([]int{}, a.shape...)
	newShape[a.ndim-2], newShape[a.ndim-1] = newShape[a.ndim-1], newShape[a.ndim-2]

	return NdArray{
		data:  out,
		shape: newShape,
		size:  a.size,
		ndim:  a.ndim,
		device: CPU,
	}
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
func (a NdArray) Device() string {
	if a.device == CPU {
		return "cpu"
	} else if a.device == CUDA {
		return "cuda"
	} else {
		return "undefined"
	}
}
// End getters
