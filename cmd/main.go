package main

import "fmt"
import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

func main() {

	batchSize := 1
	inDim     := 2
	outDim    := 3

	input := tensor.Rand([]int{batchSize, inDim})
	
	model := nn.LinearNew(inDim, outDim)
	params := model.Parameters()

	output := model.Forward(&input)
	output.Backward()

	fmt.Println(input.Print())
	for i := range params {
		fmt.Println(params[i].Print())
	}
	fmt.Println(output.Print())

	// w := tensor.Rand([]int{inDim, outDim})
	// b := tensor.Rand([]int{outDim})
	//
	// w_out := input.MatMul(&w)
	// b_out := w_out.Add(&b)
	// b_out.Backward()
	//
	// fmt.Println(input.Print())
}
