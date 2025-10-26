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

// Matrix multiplication
func (a *Tensor) MatMul(b *Tensor) Tensor {

	// Check that dimensions are compatible for mat mul
	if a.data.Shape()[1] != b.data.Shape()[0] {
		panic("NdArray matmul error: Inner dimensions must match")
	}

	backward := func(self *Tensor) {
		// Note: multiplication order is important for mat muls
		// dL/da = dL/dself ⋅ b^T 
		// dL/db = a^T ⋅ dL/dself
		a.grad = a.grad.Add(self.grad.MatMul(b.data.T()))
		b.grad = b.grad.Add(a.data.T().MatMul(self.grad))
	}

	return Tensor{
		data: a.data.MatMul(b.data),
		grad: nd.Zero([]int{a.data.Shape()[0], b.data.Shape()[1]}),
		b: backward,
		p1: a,
		p2: b,
	}
}
