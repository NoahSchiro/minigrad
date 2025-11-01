package tensor;

import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func (a *Tensor) ScalarAdd(input float32) *Tensor {

	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * 1
		a.grad = a.grad.Add(self.grad)
	}

	return &Tensor{
		data: a.data.ScalarAdd(input),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

func (a *Tensor) ScalarMul(input float32) *Tensor {
	
	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * input
		a.grad = a.grad.Add(
			self.grad.ScalarMul(input),
		)
	}

	return &Tensor{
		data: a.data.ScalarMul(input),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

func (a *Tensor) Neg() *Tensor {
	
	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * input
		a.grad = a.grad.Add(
			self.grad.Neg(),
		)
	}

	return &Tensor{
		data: a.data.Neg(),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

func (t *Tensor) Sum() *Tensor {
	// Compute forward pass
	sumData := t.data.Sum()

	// Define backward function
	backward := func(self *Tensor) {
		for i := range t.grad.Size() {
			x := t.grad.GetLinear(i) + self.grad.GetLinear(0)
			t.grad.SetLinear(i, x)
		}
	}

	return &Tensor{
		data: sumData,
		grad: nd.Zero([]int{1}),
		b:    backward,
		p1:   t,
		p2:   nil,
	}
}

func (a *Tensor) Abs() *Tensor {
	d := a.data.Abs()

	backward := func (self *Tensor) {
		for i := range a.data.Size() {

			var sign float32 = 1.0
			if a.data.GetLinear(i) < 0 {
				sign = -1.0
			}
			x := a.grad.GetLinear(i) + self.grad.GetLinear(i) * sign
			a.grad.SetLinear(i, x)
		}
	}

	return &Tensor{
		data: d,
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}

// func (a *Tensor) ReLu() Tensor {
// }
