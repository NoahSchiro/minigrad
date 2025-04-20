package main

import "fmt"
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

	a := nd.Rand([]int{2,3,2})
	b := nd.Rand([]int{3,2,2})

	a.Print()
	b.Print()
	c, err := a.Add(b)
	if err != nil {
		fmt.Println(err)
		return
	}
	c.Print()
}
