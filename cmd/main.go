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

	input := t.Rand([]int{batchSize, inDim})
	
	model := nn.LinearNew(inDim, outDim)
	params := model.Parameters()
	optim := nn.SGDNew(params, 0.1)

	output := model.Forward(&input)
	fmt.Println("Before backwards")
	printLinearLayers(params)

	output.Backward()
	
	fmt.Println("After backwards")
	printLinearLayers(params)

	optim.Update()
	
	fmt.Println("After update")
	printLinearLayers(params)

	optim.ZeroGrad()
	
	fmt.Println("After zeroing")
	printLinearLayers(params)
}
