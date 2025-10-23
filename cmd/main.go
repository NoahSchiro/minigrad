package main

import "fmt"
import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"

func main() {

	a := tensor.New([]float32{1,2}, []int{2})
	b := a.ScalarAdd(2)
	c := b.Neg()
	d := c.ScalarMul(4)

	fmt.Println("a =", a.Print())
	fmt.Println("b =", b.Print())
	fmt.Println("c =", c.Print())
	fmt.Println("d =", d.Print())

	d.Backward()

	fmt.Println("a =", a.Print())
	fmt.Println("b =", b.Print())
	fmt.Println("c =", c.Print())
	fmt.Println("d =", d.Print())
}
