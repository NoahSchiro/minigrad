package nn

import t "github.com/NoahSchiro/minigrad/pkg/tensor"

type Linear struct {
	weight t.Tensor
	bias t.Tensor
}

func LinearNew(inDim int, outDim int) Linear {
	return Linear{
		weight: t.Rand([]int{inDim, outDim}),
		bias: t.Rand([]int{outDim}),
	}
}

func (a Linear) Forward(input t.Tensor) t.Tensor {
	w_out := input.MatMul(&a.weight)
	return w_out.Add(&a.bias)
}

func (a Linear) Paramaters() []*t.Tensor {
	return []*t.Tensor{
		&a.weight,
		&a.bias,
	}
}
