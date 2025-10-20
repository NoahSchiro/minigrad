package tensor;

func (a *Tensor) ElemAdd(input float32) {
	a.data.ElemAdd(input)

	//Some gradient stuff?
}

func (a *Tensor) ElemMul(input float32) {
	a.data.ElemMul(input)
	//Some gradient stuff?
}

func (a *Tensor) Neg() {
	a.data.ElemNeg()
	//Some gradient stuff?
}

func (a *Tensor) ReLu() {
	a.data.ReLu()
}
