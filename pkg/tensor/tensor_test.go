package tensor;

import "fmt"
import "testing"

//----------- Begin Init functions -----------
func TestTensorNew(t *testing.T) {
	
	a := New([]float32{1}, []int{1})


	if a.data.Shape()[0] != 1 { t.Errorf("Shape of Tensor in New didn't work")}
	if a.data.Size() != 1 { t.Errorf("Size was not computed correctly in New") }
	if a.data.Ndim() != 1 { t.Errorf("Number of dims in New is wrong") }

	elem, err := a.data.Get([]int{0})
	if err != nil {
		fmt.Println("Error in fetching first elem of tensor")
	}
	if elem != 1 { t.Errorf("Fetching data from Tensor NdArray was wrong") }

	// More complex
	a = New(
		[]float32{
			1,0,
			0,1,
		},
		[] int{2,2},
	)

	if a.data.Shape()[0] != 2 { t.Errorf("Shape of Tensor in New didn't work")}
	if a.data.Size() != 4 { t.Errorf("Size was not computed correctly in New") }
	if a.data.Ndim() != 2 { t.Errorf("Number of dims in New is wrong") }

	elem, err = a.data.Get([]int{1,0})
	if err != nil {
		fmt.Println("Error in fetching (1,0) elem of tensor")
	}
	if elem != 0 { t.Errorf("Fetching data from Tensor NdArray was wrong") }
	
	elem, err = a.data.Get([]int{1,1})
	if err != nil {
		fmt.Println("Error in fetching (1,1) elem of tensor")
	}
	if elem != 1 { t.Errorf("Fetching data from Tensor NdArray was wrong") }
}
//----------- End Init functions -----------

//----------- Begin getter functions -----------
func TestGetters(t *testing.T) {

	input_data := []float32{
		1,0,
		0,1,
	}
	input_shape := [] int{2,2}

	a := New(input_data, input_shape)

	shape := a.Shape()
	size := a.Size()
	ndim := a.Ndim()

	for i := range shape {
		if shape[i] != input_shape[i] {
			t.Errorf("Input shape and fetched shape in Tensor getter does not match")
		}
	}
	if size != 4 { t.Errorf("Fetched size in Tensor getter does not match") }
	if ndim != 2 { t.Errorf("Fetched ndim in Tensor getter does not match") }
}

//----------- End getter functions -----------

//----------- Begin backwards functions -----------
func TestBuildTopo(t *testing.T) {
	a := NewFill(1, []int{1})
	b := NewFill(2, []int{1})
	c := NewFill(3, []int{1})
	d := NewFill(4, []int{1})

	d.p1 = &b
	d.p2 = &c
	
	b.p1 = &a
	c.p1 = &a

	// Ordering should be a, b, c, d
	//     b --> d
	//    ^     ^
	//   /     /
	// a --> c
	topo := buildTopo(&d)

	if topo[0] != &a {
		fmt.Println("Topological ordering is wrong")
	}
	if topo[1] != &b {
		fmt.Println("Topological ordering is wrong")
	}
	if topo[2] != &c {
		fmt.Println("Topological ordering is wrong")
	}
	if topo[3] != &d {
		fmt.Println("Topological ordering is wrong")
	}
}
//----------- End backwards functions -----------

//----------- Begin unary functions -----------
func TestScalarAdd(t *testing.T) {
	a := New([]float32{1,2}, []int{2})
	b := a.ScalarAdd(2)

	b.Backward()

	elem, _ := b.data.Get([]int{0})
	if elem != 3 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != 5 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
}

func TestScalarMul(t *testing.T) {
	a := New([]float32{1,2}, []int{2})
	b := a.ScalarMul(2)

	b.Backward()

	elem, _ := b.data.Get([]int{0})
	if elem != 2 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != 4 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != 2 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
}

func TestNeg(t *testing.T) {
	a := New([]float32{1,2}, []int{2})
	b := a.Neg()

	b.Backward()

	elem, _ := b.data.Get([]int{0})
	if elem != -1 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != -2 {
		fmt.Println("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != -1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != -1 {
		fmt.Println("ScalarAdd gradient incorrect")
	}
}
//----------- End unary functions -----------
//----------- Begin binary functions -----------
func TestAdd(t *testing.T) {
	a := New(
		[]float32{1,2,3,4},
		[]int{2,2},
	)
	b := New(
		[]float32{-4,-3,-2,-1},
		[]int{2,2},
	)
	c := a.Add(&b)
	c.Backward()

	elem, _ := c.data.Get([]int{0,0})
	if elem != -3 {
		fmt.Println("Add did not add correctly")
	}
	elem, _ = c.data.Get([]int{1,0})
	if elem != 1 {
		fmt.Println("Add did not add correctly")
	}

	elem, _ = c.grad.Get([]int{1,1})
	if elem != 1 {
		fmt.Println("Add gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1,0})
	if elem != 1 {
		fmt.Println("Add gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0,1})
	if elem != 1 {
		fmt.Println("Add gradient incorrect")
	}
}
//----------- End binary functions -----------

