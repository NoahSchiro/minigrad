package ndarray

func (a NdArray) Add(b NdArray) NdArray {
	
	// Check that shapes match
	if !checkShape(a, b) {
		panic("NdArray add error: Shapes must match")
	}

	data := make([]float32, a.size)

	for i := range a.data {
		data[i] = a.data[i] + b.data[i]
	}

	return NdArray{
		data: data,
		shape: a.shape,
		size: a.size,
		ndim: a.ndim,
	}
}
