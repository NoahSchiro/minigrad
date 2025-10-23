package tensor;

import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func (a *Tensor) ElemAdd(input float32) Tensor {

	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * 1
		a.grad = a.grad.Add(self.grad)
	}

	return Tensor{
		data: a.data.ElemAdd(input),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

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
