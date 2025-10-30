package tensor;

import "fmt"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

// Matrix addition
func (a *Tensor) Add(b *Tensor) Tensor {
	aShape := a.data.Shape()
	bShape := b.data.Shape()
	
	var outData nd.NdArray

	switch {
	case checkShape(*a, *b):
		// If they are the same, normal addition
		outData = a.data.Add(b.data)
	case len(aShape) == 2 && len(bShape) == 1 && aShape[1] == bShape[0]:

		// Broadcast add across the first dim if appropriate
		outData = nd.Zero(aShape)
		for i := 0; i < aShape[0]; i++ {
			for j := 0; j < aShape[1]; j++ {
				a_data, _ := a.data.Get([]int{i, j})
				b_data, _ := b.data.Get([]int{j})
				outData.Set([]int{i, j}, a_data + b_data) 
			}
		}
	default:
		panic(fmt.Sprintf("Add: incompatible shapes %v and %v", aShape, bShape))
	}

	// Define backward closure
	backward := func(self *Tensor) {
		// Gradient w.r.t a
		a.grad = a.grad.Add(self.grad)

		// Gradient w.r.t b
		if len(aShape) == 2 && len(bShape) == 1 && aShape[1] == bShape[0] {
			// Sum over batch dimension for bias
			sumGrad := nd.Zero(bShape)
			for j := 0; j < bShape[0]; j++ {
				var total float32
				total = 0
				for i := 0; i < aShape[0]; i++ {
					value, _ := self.grad.Get([]int{i,j})
					total += value
				}
				sumGrad.Set([]int{j}, total)
			}
			b.grad = b.grad.Add(sumGrad)
		} else {
			b.grad = b.grad.Add(self.grad)
		}
	}

	return Tensor{
		data: outData,
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
