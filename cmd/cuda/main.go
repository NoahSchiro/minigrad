package main

import (
	"fmt"
)
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func main() {

	x := nd.Rand([]int{2,2})

	x.To(nd.CUDA)
	y := x.Abs()

	x.To(nd.CPU)
	y.To(nd.CPU)

	fmt.Println(x.Print())
	fmt.Println(y.Print())
}
