package ndarray

func (a *NdArray) UnaryApply(f func(float32) float32) {
	for i := range a.data {
		a.data[i] = f(a.data[i])
	}
}

// Add the input to each element
func (a *NdArray) ElemAdd(input float32) {
	a.UnaryApply(func(x float32) float32 {
		return x+input
	})
}

// Add the input to each element
func (a *NdArray) ElemMul(input float32) {
	a.UnaryApply(func(x float32) float32 {
		return x*input
	})
}

// Flip the sign
func (a *NdArray) Neg() {
	a.UnaryApply(func(x float32) float32 {
		return x * -1
	})
}

func (a *NdArray) ReLu() {
	a.UnaryApply(func(x float32) float32 {
		if x > 0 {
			return x
		} else {
			return 0
		}
	})
}

