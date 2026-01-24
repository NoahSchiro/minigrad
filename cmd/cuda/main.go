package main

import "fmt"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func main() {

	x := nd.Rand([]int{10,10})

	x.To(nd.CUDA)

	fmt.Printf("x is on %s\n", x.Device())
}
