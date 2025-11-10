package main

import (
	"fmt"
	"math"
	"time"
)
import t "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

const LR float32 = 0.01
const EPOCHS int = 5000 
const TRIALS int = 100

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

func train() *t.Tensor {

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
	//optim := nn.SGDNew(model.Parameters(), LR)
	optim := nn.AdamNew(
		model.Parameters(),
		LR,
		0.9, 0.999, 1e-8,
	)

	// Train
	for range EPOCHS {
		out := model.Forward(&x)
		//loss := nn.AbsErr(out, &y)
		loss := nn.MSE(out, &y)
		loss.Backward()
		optim.Update()
		optim.ZeroGrad()
	}

	// Test
	return model.Forward(&x)
}

func main() {

	fails := 0

	start := time.Now()
	for range TRIALS {
		out := train()
		y_pred0 := math.Round(float64(out.GetLinear(0)))
		y_pred1 := math.Round(float64(out.GetLinear(1)))
		y_pred2 := math.Round(float64(out.GetLinear(2)))
		y_pred3 := math.Round(float64(out.GetLinear(3)))

		if y_pred0 != 0 || y_pred1 != 1 || y_pred2 != 1 || y_pred3 != 0 {
			fails += 1
		}
	}
	duration := time.Since(start)

	fail_rate := float64(fails) / float64(TRIALS)

	fmt.Println("Fail rate", fail_rate)
	if fail_rate > 0.1 {
		fmt.Println("Failed to converge 90% of the time")
	} else {
		fmt.Printf("Duration was %f ms per trial\n", float64(duration.Milliseconds()) / float64(TRIALS))
	}
}
