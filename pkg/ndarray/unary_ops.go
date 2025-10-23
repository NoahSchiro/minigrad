package ndarray

func (a NdArray) UnaryApply(f func(float32) float32) NdArray {

	// Create a copy
	result := a.Clone()

	for i := range result.data {
		result.data[i] = f(a.data[i])
	}

	return result
}

// Add the input to each element
func (a NdArray) ScalarAdd(input float32) NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return x+input
	})
}

// Add the input to each element
func (a NdArray) ScalarMul(input float32) NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return x*input
	})
}

// Flip the sign
func (a NdArray) Neg() NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return x * -1
	})
}

func (a NdArray) ReLu() NdArray {
	return a.UnaryApply(func(x float32) float32 {
		if x > 0 {
			return x
		} else {
			return 0
		}
	})
}

