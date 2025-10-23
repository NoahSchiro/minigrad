package tensor;

// func (a *Tensor) ElemAdd(input float32) Tensor {
// }

func (a *Tensor) ElemMul(input float32) {
	a.data.ElemMul(input)
	//Some gradient stuff?
}

func (a *Tensor) Neg() {
	a.data.Neg()
	//Some gradient stuff?
}

func (a *Tensor) ReLu() {
	a.data.ReLu()
}
