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
