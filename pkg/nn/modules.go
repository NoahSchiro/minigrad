package nn

import (
	"math"
	"math/rand"
	"time"
)
import t "github.com/NoahSchiro/minigrad/pkg/tensor"

type Linear struct {
	weight t.Tensor
	bias t.Tensor
}

func LinearNew(inDim int, outDim int) *Linear {
	rand.Seed(time.Now().UnixNano())

	// Xavier initialization with zero bias
	limit := float32(math.Sqrt(6.0 / float64(inDim+outDim)))
	w := t.Uniform([]int{inDim, outDim}, -limit, limit)
	b := t.Zero([]int{outDim})

	return &Linear{
		weight: *w,
		bias: *b,
	}
}

func (a *Linear) Forward(input *t.Tensor) *t.Tensor {
	w_out := input.MatMul(&a.weight)
	b_out := w_out.Add(&a.bias)
	return &b_out
}

func (a *Linear) Parameters() []*t.Tensor {
	return []*t.Tensor{
		&a.weight,
		&a.bias,
	}
}
