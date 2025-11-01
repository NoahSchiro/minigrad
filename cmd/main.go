package main

import "fmt"
import t "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

func printLinearLayers(layers []*t.Tensor) {
	for i := range layers {
		fmt.Println(layers[i].Print())
	}
}

func main() {

	batchSize := 1
	inDim     := 2
	outDim    := 3

	x := t.Rand([]int{batchSize, inDim})
	y := t.NewFill(3, []int{1, 3})
	
	model := nn.LinearNew(inDim, outDim)
	params := model.Parameters()
	optim := nn.SGDNew(params, 0.1)

	for range 100 {
		y_pred := model.Forward(&x)
		loss := nn.AbsErr(&y_pred, &y)
		loss.Backward()
		optim.Update()
		optim.ZeroGrad()

		fmt.Println(loss.Print())
	}
}
