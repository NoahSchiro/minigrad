package ndarray

func (a *NdArray) UnaryApply(f func(float32) float32) {
	for i := range a.data {
		a.data[i] = f(a.data[i])
	}
}
