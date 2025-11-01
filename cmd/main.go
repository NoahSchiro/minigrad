package main

import "fmt"
import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

func main() {

	batchSize := 1
	inDim     := 8
	outDim    := 5

	input := tensor.Rand([]int{batchSize, inDim})
	fmt.Println(input.Shape())

	model := nn.LinearNew(inDim, outDim)

	output := model.Forward(input)
	fmt.Println(output.Shape())
}
