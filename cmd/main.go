package main

import "fmt"
import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"

func main() {

	a := tensor.New([]float32{1}, []int{1})
	b := a.ElemAdd(1)
	c := b.ElemAdd(1)
	d := c.ElemAdd(1)

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
