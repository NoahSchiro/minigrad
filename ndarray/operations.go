package ndarray

import "errors"

func (a NdArray) Add(b NdArray) (NdArray, error) {
	// Check that shapes match
	empty := Empty()

	if !checkShape(a, b) {
		return empty, errors.New("Error: Shapes must match on add operation")
	}

	data := make([]float32, a.size)

	for i := range a.data {
		data[i] = a.data[i] + b.data[i]
	}

	return NdArray{
		data: data,
		shape: a.shape,
		size: a.size,
	}, nil
}

func (a *NdArray) UnaryApply(f func(float32) float32) {
	for i := range a.data {
		a.data[i] = f(a.data[i])
	}
}
