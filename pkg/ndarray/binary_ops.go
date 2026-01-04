package ndarray

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#cgo LDFLAGS: -L${SRCDIR}/cuda -lcuda
#include "vector.h"
*/
import "C"
import "unsafe"

import (
	"runtime"
	"sync"
)

// Matrix addition
// Serial or parallel depending on size
// CPU or CUDA depending on a.device
func (a NdArray) Add(b NdArray) NdArray {
	// Check that shapes match
	if !checkShape(a, b) {
		panic("NdArray add error: Shapes must match")
	}

	// Computation can only happen if they
	// are on the same device
	if a.device != b.device {
		panic("NdArray add error: Arrays must be on same device")
	}

	if a.device == CPU {
		// Experimentally, the parallel is only
		// worth it above 4k elements.
		if a.size > 4_000 {
			return a.addParallel(b)
		} else {
			return a.addSerial(b)
		}
	} else if a.device == CUDA {
		return a.addCuda(b)
	} else {
		panic("NdArray add error: Device of array not recognized")
	}

}

// Matrix addition (serially)
func (a NdArray) addSerial(b NdArray) NdArray {
	result := a.Clone()
	for i := range result.data {
		result.data[i] += b.data[i]
	}
	return result
}

// Matrix addition (parallel)
func (a NdArray) addParallel(b NdArray) NdArray {
	result := a.Clone()

	// How much work, how many workers do we
	// have, and how much work should we give each one?
	n := len(result.data)
	workers := runtime.GOMAXPROCS(0)
	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := range workers {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}

		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				result.data[i] += b.data[i]
			}
		}(start, end)
	}

	wg.Wait()
	return result
}

// Add a and b assuming they are on device,
// and put answer on device
func (a NdArray) addCuda(b NdArray) NdArray {
	d_data := C.cuda_vector_add(
		(*C.float)(a.gpuData),
		(*C.float)(b.gpuData),
		(C.int)(a.size),
	)

	return NdArray{
		data: nil,
		gpuData: unsafe.Pointer(d_data),
		shape: a.shape,
		size: a.size,
		ndim: a.ndim,
		device: CUDA,
	}
}

// Element wise matrix multiplication
func (a NdArray) ElemMul(b NdArray) NdArray {
	if !checkShape(a, b) {
		panic("NdArray ElemAdd error: shapes must match")
	}

	result := a.Clone()

	for i := range b.data {
		result.data[i] *= b.data[i]
	}
	return result
}

// Matrix multiplication
func (a NdArray) MatMul(b NdArray) NdArray {
	// Validate minimum dimensions
	if a.ndim < 2 || b.ndim < 2 {
		panic("NdArray MatMul: Inputs must have at least 2 dimensions")
	}

	// Get matrix dimensions
	aCols := a.shape[a.ndim-1]
	bRows := b.shape[b.ndim-2]
	if aCols != bRows {
		panic("NdArray MatMul: Incompatible inner dimensions")
	}

	// Validate leading dimensions (must be equal)
	aLeading := a.shape[:a.ndim-2]
	bLeading := b.shape[:b.ndim-2]

	if len(aLeading) != len(bLeading) {
		panic("NdArray MatMul mismatched leading dimensions")
	}
	for i := range aLeading {
		if aLeading[i] != bLeading[i] {
			panic("NdArray MatMul mismatched leading dimensions")
		}
	}

	// Experimentally, the parallel is only
	// worth it above 3k elements.
	if a.size > 3_000 {
		return a.matMulParallel(b)
	} else {
		return a.matMulSerial(b)
	}
}

// Matrix multiplication (serial)
func (a NdArray) matMulSerial(b NdArray) NdArray {
	aCols := a.shape[a.ndim-1]
	bRows := b.shape[b.ndim-2]
	aLeading := a.shape[:a.ndim-2]

	// Calculate total number of matrices 
	totalMatrices := 1
	for i := range aLeading {
		totalMatrices *= aLeading[i]
	}
	aRows := a.shape[a.ndim-2]
	bCols := b.shape[b.ndim-1]

	// Prepare result array
	resultShape := append(append([]int{}, aLeading...), aRows, bCols)
	resultSize := 1
	for i := range resultShape {
		resultSize *= resultShape[i]
	}
	resultData := make([]float32, resultSize)

	// Matrix sizes
	aMatrixSize := aRows * aCols
	bMatrixSize := bRows * bCols

	// Batch multiplication
	for m := 0; m < totalMatrices; m++ {
		offsetA := m * aMatrixSize
		offsetB := m * bMatrixSize
		offsetRes := m * aRows * bCols

		for i := 0; i < aRows; i++ {
			for j := 0; j < bCols; j++ {
				var sum float32
				for p := 0; p < aCols; p++ {
					sum += a.data[offsetA+i*aCols+p] * 
                            b.data[offsetB+p*bCols+j]
				}
				resultData[offsetRes+i*bCols+j] = sum
			}
		}
	}

	return NdArray{
		data:  resultData,
		shape: resultShape,
		size:  resultSize,
		ndim:  len(resultShape),
	}
}

func (a NdArray) matMulParallel(b NdArray) NdArray {
	aCols := a.shape[a.ndim-1]
	bRows := b.shape[b.ndim-2]
	
	aLeading := a.shape[:a.ndim-2]

	// Calculate total number of matrices 
	totalMatrices := 1
	for _, x := range aLeading {
		totalMatrices *= x
	}

	aRows := a.shape[a.ndim-2]
	bCols := b.shape[b.ndim-1]

	// Prepare result array
	resultShape := append(append([]int{}, aLeading...), aRows, bCols)
	resultSize := 1
	for _, x := range resultShape {
		resultSize *= x
	}
	resultData := make([]float32, resultSize)

	// Matrix sizes
	aMatrixSize := aRows * aCols
	bMatrixSize := bRows * bCols

	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup

	// Split by rows for single matrices,
	// or by batches if multiple
	for m := range totalMatrices {
		offsetA := m * aMatrixSize
		offsetB := m * bMatrixSize
		offsetRes := m * aRows * bCols

		chunk := (aRows + workers - 1) / workers
		wg.Add(workers)

		for w := 0; w < workers; w++ {
			start := w * chunk
			end := start + chunk
			if end > aRows {
				end = aRows
			}

			go func(s, e int) {
				defer wg.Done()
				for i := s; i < e; i++ {
					for j := 0; j < bCols; j++ {
						var sum float32
						for p := 0; p < aCols; p++ {
							sum += a.data[offsetA+i*aCols+p] * b.data[offsetB+p*bCols+j]
						}
						resultData[offsetRes+i*bCols+j] = sum
					}
				}
			}(start, end)
		}
	}

	wg.Wait()

	return NdArray{
		data:  resultData,
		shape: resultShape,
		size:  resultSize,
		ndim:  len(resultShape),
	}
}
