package main

import "fmt"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"
//import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"

func main() {

	a := nd.New([]float32{1.,2.,3.,4.}, []int{2,2})

	fmt.Println(a.Print())
	elem, err := a.Get([]int{0,0})

	fmt.Println(err)
	fmt.Println(elem)
}
