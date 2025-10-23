package tensor;

import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

// Matrix addition
func (a *Tensor) Add(b *Tensor) Tensor {
	
	// Check that shapes match
	if !checkShape(*a, *b) {
		panic("NdArray add error: Shapes must match")
	}

	backward := func(self *Tensor) {
		// dL/da = dL/db = dL/dself * 1
		a.grad = a.grad.Add(self.grad)
		b.grad = b.grad.Add(self.grad)
	}

	return Tensor{
		data: a.data.Add(b.data),
		grad: nd.Zero(a.data.Shape()),
		b: backward,
		p1: a,
		p2: b,
	}
}


