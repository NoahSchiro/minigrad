package main

import "fmt"
import t "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

const LR float32 = 0.01
const EPOCHS int = 15000

type Model struct {
	l1 *nn.Linear
	l2 *nn.Linear
}
func ModelNew() *Model {
	return &Model{
		l1: nn.LinearNew(2,2),
		l2: nn.LinearNew(2,1),
	}
}
func (a *Model) Forward(input *t.Tensor) *t.Tensor {
	l1_out := a.l1.Forward(input).Sigmoid()
	l2_out := a.l2.Forward(l1_out)
	return l2_out
}
func (a *Model) Parameters() []*t.Tensor {
	l1p := a.l1.Parameters()
	l2p := a.l2.Parameters()
	concat := append(l1p, l2p...)
	return concat
}

func main() {

	// Define data for the xor problem
	x := t.New(
		[]float32{
			0,0,
			0,1,
			1,0,
			1,1,
		},
		[]int{4,2},
	)
	y := t.New(
		[]float32{
			0,
			1,
			1,
			0,
		},
		[]int{4,1},
	)

	// Model and optimizer
	model := ModelNew()
	optim := nn.SGDNew(model.Parameters(), LR)

	// Train
	for range EPOCHS {
		out := model.Forward(&x)
		loss := nn.AbsErr(out, &y)
		loss.Backward()
		optim.Update()
		optim.ZeroGrad()
	}

	// Test
	out := model.Forward(&x)
	fmt.Println(out.Print())
}
