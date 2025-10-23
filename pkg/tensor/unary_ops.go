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

func (a *Tensor) ElemMul(input float32) Tensor {
	
	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * input
		a.grad = a.grad.Add(
			self.grad.ElemMul(input),
		)
	}

	return Tensor{
		data: a.data.ElemMul(input),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

func (a *Tensor) Neg() Tensor {
	
	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * input
		a.grad = a.grad.Add(
			self.grad.Neg(),
		)
	}

	return Tensor{
		data: a.data.Neg(),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

// func (a *Tensor) ReLu() Tensor {
// }
