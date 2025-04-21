package ndarray

import "testing"


//----------- Begin Init functions -----------
func TestNdArrayNew(t *testing.T) {

	a := New([]float32{1}, []int{1})

	if a.data[0] != 1 { t.Errorf("Passing data to New didn't work") }
	if a.shape[0] != 1 { t.Errorf("Shape of NdArray in New didn't work")}
	if a.size != 1 { t.Errorf("Size was not computed correctly in New") }
	if a.ndim != 1 { t.Errorf("Number of dims in New is wrong") }

	// More complex
	b := New(
		[]float32{
			1,0,
			0,1,
		},
		[] int{2,2},
	)
	
	if b.ndim != 2 { t.Errorf("Number of dims in New is wrong") }
	if b.data[3] != 1 && b.data[2] != 0 { t.Errorf("Passing data to New didn't work") }
	if b.shape[0] != 2 { t.Errorf("Shape of NdArray in New didn't work")}
	if b.size != 4 { t.Errorf("Size was not computed correctly in New") }
}

func TestNdArrayRand(t *testing.T) {
	a := Rand([]int{5,4,3})
	
	if a.shape[0] != 5 { t.Errorf("Shape of NdArray in Rand didn't work")}
	if a.shape[1] != 4 { t.Errorf("Shape of NdArray in Rand didn't work")}
	if a.shape[2] != 3 { t.Errorf("Shape of NdArray in Rand didn't work")}
	if a.size != 5*4*3 { t.Errorf("Size was not computed correctly in Rand") }
	if a.ndim != 3 { t.Errorf("Number of dims in Rand is wrong") }
}

func TestNdArrayFill(t *testing.T) {
	a := NewFill(-1, []int{5,4,3})
	
	if a.shape[0] != 5 { t.Errorf("Shape of NdArray in NewFill didn't work")}
	if a.shape[1] != 4 { t.Errorf("Shape of NdArray in NewFill didn't work")}
	if a.shape[2] != 3 { t.Errorf("Shape of NdArray in NewFill didn't work")}
	if a.size != 5*4*3 { t.Errorf("Size was not computed correctly in NewFill") }
	if a.ndim != 3 { t.Errorf("Number of dims in NewFill is wrong") }
	if a.data[30] != -1 { t.Errorf("Data fill was wrong in NewFill") }
}

func TestEmpty(t *testing.T) {
	a := Empty()
	
	if a.size != 1 { t.Errorf("Empty Init failed (wrong size)") }
	if a.ndim != 1 { t.Errorf("Empty Init failed (wrong ndim)") }
	if a.data[0] != 0 { t.Errorf("Empty Init failed (wrong data)") }
	if a.shape[0] != 1 { t.Errorf("Empty Init failed (wrong shape)") }
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
			t.Errorf("Input shape and fetched shape in NdArray getter does not match")
		}
	}
	if size != 4 { t.Errorf("Fetched size in NdArray getter does not match") }
	if ndim != 2 { t.Errorf("Fetched ndim in NdArray getter does not match") }
}
