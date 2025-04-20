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

	n, err := nd.New(data, []int{2, 3, 2})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(n.Data())
}
