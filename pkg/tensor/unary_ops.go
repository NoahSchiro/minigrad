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

func (a *Tensor) ScalarDiv(input float32) *Tensor {

	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = (dL/dself) / input
		a.grad = a.grad.Add(
			self.grad.ScalarDiv(input),
		)
	}

	return &Tensor{
		data: a.data.ScalarDiv(input),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}



func (a *Tensor) ScalarPow(pow float32) *Tensor {
	
	// Define the backward function
	backward := func(self *Tensor) {
		// dL/da = dL/dself * power * a^(power-1)
		gradTerm := a.data.ScalarPow(pow - 1).ScalarMul(pow)
		
		for i := range a.grad.Size() {
			x := a.grad.GetLinear(i) + self.grad.GetLinear(i) * gradTerm.GetLinear(i)
			a.grad.SetLinear(i, x)
		}
	}

	return &Tensor{
		data: a.data.ScalarPow(pow),
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

func (a *Tensor) Sum() *Tensor {
	// Compute forward pass
	sumData := a.data.Sum()

	// Define backward function
	backward := func(self *Tensor) {
		for i := range a.grad.Size() {
			x := a.grad.GetLinear(i) + self.grad.GetLinear(0)
			a.grad.SetLinear(i, x)
		}
	}

	return &Tensor{
		data: sumData,
		grad: nd.Zero([]int{1}),
		b:    backward,
		p1:   a,
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
			} else if a.data.GetLinear(i) == 0. {
				sign = 0.
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

func (a *Tensor) Sigmoid() *Tensor {
	d := a.data.Sigmoid()

	backward := func(self *Tensor) {
		// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
		for i := range a.data.Size() {
			y := d.GetLinear(i)
			grad := self.grad.GetLinear(i) * y * (1 - y)
			x := a.grad.GetLinear(i) + grad
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

func (a *Tensor) ReLu() *Tensor {
	d := a.data.ReLu()

	backward := func(self *Tensor) {
		// d/dx relu(x) =  1 if x >= 0
		// d/dx relu(x) =  0 if x <  0
		for i := range a.data.Size() {
			y := a.GetLinear(i)
			var grad float32 = 0
			if y > 0 {
				grad = self.grad.GetLinear(i)
			} else {
				grad = 0
			}
			x := a.grad.GetLinear(i) + grad
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

func (a *Tensor) Softmax(axis int) *Tensor {

	backward := func(self *Tensor) {
        // Allocate dx
        dx := nd.Zero(self.p1.Shape())

        // Shape parameters
        shape := self.data.Shape()     // same as x.data
        ndim := len(shape)
        dim := shape[axis]

        // Precompute index combos excluding axis
        combos := nd.AllIndexCombos(shape, axis)
        idxs1 := make([]int, ndim)
        idxs2 := make([]int, ndim)

        // For each slice of the softmax
        for _, combo := range combos {
            ci := 0
            for d := 0; d < ndim; d++ {
                if d == axis {
                    idxs1[d] = 0
                    idxs2[d] = 0
                } else {
                    idxs1[d] = combo[ci]
                    idxs2[d] = combo[ci]
                    ci++
                }
            }

            // Apply softmax gradient formula:
            // dL/dx_i = sum_j (delta_ij - y_j) * y_i * dL/dy_j

            for i := 0; i < dim; i++ {
                idxs1[axis] = i
                yi := self.data.Get(idxs1)

                var grad float32 = 0

                for j := 0; j < dim; j++ {
                    idxs2[axis] = j
                    yj := self.data.Get(idxs2)
                    dldyj := self.grad.Get(idxs2)

                    if i == j {
                        grad += (1 - yj) * yi * dldyj
                    } else {
                        grad += (-yj) * yi * dldyj
                    }
                }

                dx.Set(idxs1, grad)
            }
        }

        // Accumulate into parent gradient
        self.p1.grad = self.p1.grad.Add(dx)

	}

	return &Tensor {
		data: a.data.Softmax(axis),
		grad: nd.Zero(a.Shape()),
		b: backward,
		p1: a,
		p2: nil,
	}
}
