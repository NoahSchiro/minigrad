package main

import "fmt"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func main() {

	x := nd.NewFill(1., []int{2,2})

	fmt.Println(x.Print())
	fmt.Println(x.Device())

	x.To(nd.CUDA)

	fmt.Println(x.Print())
	fmt.Println(x.Device())

	x.To(nd.CPU)
	
	fmt.Println(x.Print())
	fmt.Println(x.Device())

}
