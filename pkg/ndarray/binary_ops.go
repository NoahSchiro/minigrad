package ndarray

// Matrix addition
func (a NdArray) Add(b NdArray) NdArray {
	
	// Check that shapes match
	if !checkShape(a, b) {
		panic("NdArray add error: Shapes must match")
	}

	result := a.Clone()

	for i := range result.data {
		result.data[i] += b.data[i]
	}

	return result
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
		panic("MatMul: Inputs must have at least 2 dimensions")
	}

	// Get matrix dimensions
	aCols := a.shape[a.ndim-1]
	bRows := b.shape[b.ndim-2]
	if aCols != bRows {
		panic("MatMul: Incompatible inner dimensions")
	}

	// Validate leading dimensions (must be equal)
	aLeading := a.shape[:a.ndim-2]
	bLeading := b.shape[:b.ndim-2]

	if len(aLeading) != len(bLeading) {
		panic("MatMul mismatched leading dimensions")
	}
	for i := range aLeading {
		if aLeading[i] != bLeading[i] {
			panic("MatMul mismatched leading dimensions")
		}
	}

	// Calculate dimensions
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
