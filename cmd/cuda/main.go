package main

import "fmt"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func main() {

	x := nd.Rand([]int{10,10})
	
	fmt.Printf(x.Print())
	fmt.Printf("x is on %s\n", x.Device())

	x.To(nd.CUDA)

	fmt.Printf(x.Print())
	fmt.Printf("x is on %s\n", x.Device())

	x.To(nd.CPU)
	fmt.Printf(x.Print())
	fmt.Printf("x is on %s\n", x.Device())
}
