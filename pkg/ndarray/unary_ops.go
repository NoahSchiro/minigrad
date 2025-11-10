package ndarray

import "math"

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

// Mul the input to each element
func (a NdArray) ScalarMul(input float32) NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return x*input
	})
}

// Div the input to each element
func (a NdArray) ScalarDiv(input float32) NdArray {
	if input == 0. {
		panic("Div by zero not allowed!")
	}
	
	return a.UnaryApply(func(x float32) float32 {
		return x/input
	})
}

func (a NdArray) ScalarPow(pow float32) NdArray {
	return a.UnaryApply(func(x float32) float32 {
		base := float64(x)
		powF64 := float64(pow)
		return float32(math.Pow(base, powF64))
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

func (a NdArray) Sum() NdArray {
	var total float32 = 0.0
	for _, v := range a.data {
		total += v
	}
	return NdArray{
		data: []float32{total},
		shape: []int{1},
		size: 1,
		ndim: 1,
	}
}

func (a NdArray) Abs() NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return float32(math.Abs(float64(x)))
	})
}

func (a NdArray) Sigmoid() NdArray {
	return a.UnaryApply(func(x float32) float32 {
		return 1 / (1 + float32(math.Exp(-float64(x))))
	})
}
