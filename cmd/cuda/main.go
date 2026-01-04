package main

import "fmt"
import "time"
import nd "github.com/NoahSchiro/minigrad/pkg/ndarray"

func main() {

	x := nd.Rand([]int{1000,1000})
	y := nd.Rand([]int{1000,1000})

    cpu_start := time.Now()
	a := x.Add(y)
    cpu_elapsed := time.Since(cpu_start)
	fmt.Printf("CPU elapsed time: %d\n", cpu_elapsed.Nanoseconds())

	x.To(nd.CUDA)
	y.To(nd.CUDA)

    cuda_start := time.Now()
	b := x.Add(y)
    cuda_elapsed := time.Since(cuda_start)
	fmt.Printf("CUDA elapsed time: %d\n", cuda_elapsed.Nanoseconds())

	b.To(nd.CPU)

	for i := range a.Size() {
		if a.GetLinear(i) != b.GetLinear(i) {
			fmt.Println("A and B do not match")
		}
	}
	fmt.Println("A and B do match")

}
