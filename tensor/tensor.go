package tensor

import nd "github.com/NoahSchiro/minigrad/ndarray"

type Tensor struct {
	data nd.NdArray      // Content
	grad nd.NdArray      // Gradients
	b func(self *Tensor) // Backwards pass
	p1 *Tensor           // Parent 1 (if applicable)
	p2 *Tensor           // Parent 2 (if applicable)
}

// Create an empty tensor
func Empty() Tensor {
	return Tensor{
		data: nd.Empty(),
		grad: nd.Empty(),
		b: func(s *Tensor) {},
		p1: nil,
		p2: nil,
	}
}

// Given data and a shape, construct a new tensor
func New(data []float32, shape []int) Tensor {
	return Tensor{
		data: nd.New(data, shape),
		grad: nd.Zero(shape),
		b: func(s *Tensor) {},
		p1: nil,
		p2: nil,
	}
}

// Given a number and a shape, create an tensor filled with that number
func NewFill(data float32, shape []int) Tensor {
	return Tensor{
		data: nd.NewFill(data, shape),
		grad: nd.Zero(shape),
		b: func(s *Tensor) {},
		p1: nil,
		p2: nil,
	}
}

// Return a zeroed out tensor of specified shape
func Zero(shape []int) Tensor {
	return Tensor{
		data: nd.Zero(shape),
		grad: nd.Zero(shape),
		b: func(s *Tensor) {},
		p1: nil,
		p2: nil,
	}
}

// Given a shape, init a random Tensor with floats in range [0,1]
func Rand(shape []int) Tensor {
	return Tensor{
		data: nd.Rand(shape),
		grad: nd.Zero(shape),
		b: func(s *Tensor) {},
		p1: nil,
		p2: nil,
	}
}

// Return the tensor as a string, for debugging
func (a Tensor) Print() string {
	s := "Tensor(\n  "
	s += a.data.Print()
	s += "\n)"
	return s
}
