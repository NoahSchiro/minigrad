package main

//import "fmt"
import nd "github.com/NoahSchiro/minigrad/ndarray"

func main() {

	shape := []int{2,2,2}
	data_a := []float32{
		1,0,
		0,1,

		4,2,
		2,4,
	}

	a := nd.New(data_a, shape)

	data_b := []float32{
		2,2,
		2,2,

		2,2,
		2,2,
	}

	b := nd.New(data_b, shape)

	a.Print()
	b.Print()

	c := a.MatMul(b)

	c.Print()
}
