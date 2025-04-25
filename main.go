package main

import "fmt"
import tensor "github.com/NoahSchiro/minigrad/tensor"

func main() {

	shape := []int{2,2,2}
	data_a := []float32{
		1,0,
		0,1,

		4,2,
		2,4,
	}

	a := tensor.New(data_a, shape)
	emp := tensor.Empty()
	newFill := tensor.NewFill(2, shape)
	zero := tensor.Zero(shape)
	rand := tensor.Rand(shape)

    fmt.Println(a.Print())
    fmt.Println(emp.Print())
    fmt.Println(newFill.Print())
    fmt.Println(zero.Print())
    fmt.Println(rand.Print())
}
