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
//----------- End getter functions -----------

//----------- Begin Utils functions -----------

func TestCheckShape(t *testing.T) {
	a := Rand([]int{1,2})
	b := Rand([]int{1,2})
	c := Rand([]int{2,1})

	if !checkShape(a,b) {
		t.Errorf("checkShape returned false when it should be true")
	}
	if checkShape(a,c) {
		t.Errorf("checkShape returned true when it should be false")
	}
}
//intArrayProduct(data []int)

func TestIntArrayProduct(t *testing.T) {
	a := intArrayProduct([]int {1,2,3,4,5})
	if a != 120 {
		t.Errorf("intArrayProduct failed on test A")
	}
	
	b := intArrayProduct([]int {34, 56})
	if b != 1904 {
		t.Errorf("intArrayProduct failed on test B")
	}
	
	c := intArrayProduct([]int {5,5})
	if c != 25 {
		t.Errorf("intArrayProduct failed on test c")
	}
}
//----------- End Utils functions -----------

//----------- Begin Unary Ops -----------
func TestElemAdd(t *testing.T) {
	// Should be all -1
	a := NewFill(-1, []int{5,4,3})
	a.ElemAdd(1)
	for i := range a.data {
		if a.data[i] != 0 {
			t.Errorf("ElemAdd did not add correctly")
		}
	}

	// Adding a negative num is our subtraction
	a.ElemAdd(-2)
	for i := range a.data {
		if a.data[i] != -2 {
			t.Errorf("ElemAdd did not subtract correctly")
		}
	}
}

func TestElemMul(t *testing.T) {
	a := NewFill(10, []int{5,4,3})
	a.ElemMul(2)
	for i := range a.data {
		if a.data[i] != 20 {
			t.Errorf("ElemMul did not mul correctly")
		}
	}

	// Adding a negative num is our subtraction
	a.ElemMul(0)
	for i := range a.data {
		if a.data[i] != 0 {
			t.Errorf("ElemMul did not mul correctly")
		}
	}
}

func TestNeg(t *testing.T) {
	a := NewFill(-1, []int{5,4,3})
	a.Neg()
	for i := range a.data {
		if a.data[i] != 1 {
			t.Errorf("Neg did not flip the sign correctly")
		}
	}

	// Adding a negative num is our subtraction
	a.Neg()
	for i := range a.data {
		if a.data[i] != -1 {
			t.Errorf("Neg did not flip the sign correctly")
		}
	}
}

func TestReLu(t *testing.T) {
	a := NewFill(-1, []int{2,2})

	a.ReLu()
	for i := range a.data {
		if a.data[i] != 0 {
			t.Errorf("ReLu did not change a negative number")
		}
	}

	b := NewFill(1, []int{2,2})
	for i := range b.data {
		if b.data[i] != 1 {
			t.Errorf("ReLu changed a positive number")
		}
	}
}
//----------- End Unary Ops -----------
