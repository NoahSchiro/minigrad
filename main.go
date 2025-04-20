package main

//import "fmt"
import nd "github.com/NoahSchiro/minigrad/ndarray"

func main() {

	data := make([]float32, 12)

	for i := range data {
		if i < 5 {
			data[i] = 0.
		} else {
			data[i] = 1.
		}
	}

	n := nd.Rand([]int{2,3,2})

	n.Print()

	n.UnaryApply(func(x float32) float32 {
		return x * -1
	})

	n.Print()
}
