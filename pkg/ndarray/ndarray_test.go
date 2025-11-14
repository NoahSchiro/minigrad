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

func TestNdArrayNewFill(t *testing.T) {
	shape := []int{5,4,3}
	a := NewFill(-1, shape)
	
	if a.shape[0] != 5 {
		t.Errorf("Shape of NdArray in NewFill didn't work")
	}
	if a.shape[1] != 4 {
		t.Errorf("Shape of NdArray in NewFill didn't work")
	}
	if a.shape[2] != 3 {
		t.Errorf("Shape of NdArray in NewFill didn't work")
	}
	if a.size != 5*4*3 {
		t.Errorf("Size was not computed correctly in NewFill")
	}
	if a.ndim != 3 {
		t.Errorf("Number of dims in NewFill is wrong")
	}
	if a.data[30] != -1 {
		t.Errorf("Data fill was wrong in NewFill")
	}
	shape[1] = 42
	if a.shape[1] == 42 {
		t.Errorf("Two objects are sharing memory")
	}
}

func TestEmpty(t *testing.T) {
	a := Empty()
	
	if a.size != 1 { t.Errorf("Empty Init failed (wrong size)") }
	if a.ndim != 1 { t.Errorf("Empty Init failed (wrong ndim)") }
	if a.data[0] != 0 { t.Errorf("Empty Init failed (wrong data)") }
	if a.shape[0] != 1 { t.Errorf("Empty Init failed (wrong shape)") }
}

func TestClone(t *testing.T) {
	a := New([]float32{1}, []int{1})
	b := a.Clone()

	b.data[0] = 4

	if b.data[0] != 4 {
		t.Errorf("Shouldn't be possible")
	}

	if a.data[0] != 1 {
		t.Errorf("NdArray A should be unaffected by clone")
	}
}

func TestNdArrayFill(t *testing.T) {

	a := Rand([]int{3,3,3})
	a.Fill(1.0)

	for elem := range a.data {
		if a.data[elem] != 1.0 {
			t.Errorf("Filling an initialized array did not work")
		}
	}
}
//----------- End Init functions -----------
//----------- Begin Shape functions -----------
func TestNdArrayTranspose(t *testing.T) {
	a := New([]float32{1,2,3,4}, []int{2,2})

	b := a.T()

	if b.data[0] != 1 {
		t.Errorf("Transpose didn't work on matrix")
	}
	if b.data[1] != 3 {
		t.Errorf("Transpose didn't work on matrix")
	}
	if b.data[2] != 2 {
		t.Errorf("Transpose didn't work on matrix")
	}
	if b.data[3] != 4 {
		t.Errorf("Transpose didn't work on matrix")
	}

	x := New(
		[]float32{
			1, 2, 3,
			4, 5, 6,

			7, 8, 9,
			10,11,12,

			13,14,15,
			16,17,18,
		},
		[]int{3, 2, 3},
	)

	y := x.T()
	elem := y.Get([]int{0,2,1})
	if elem != 6 {
		t.Errorf("Transpose didn't work on tensor")
	}
	elem = y.Get([]int{1,1,1})
	if elem != 11 {
		t.Errorf("Transpose didn't work on tensor")
	}
	elem = y.Get([]int{2,0,1})
	if elem != 16 {
		t.Errorf("Transpose didn't work on tensor")
	}
}
//----------- End Shape functions -----------
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

func TestNdArrayGet(t *testing.T) {

	// 1,2,
	// 3,4
	a := New([]float32{1.,2.,3.,4.}, []int{2,2})

	// Try to get an answer
	ans := a.Get([]int{1,1})
	if ans != 4.0 {
		t.Errorf("Get answer was wrong. Got %f, expected 4.0", ans)
	}
	// Try another
	ans = a.Get([]int{0,0})
	if ans != 1.0 {
		t.Errorf("Get answer was wrong. Got %f, expected 1.0", ans)
	}
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
func TestScalarAdd(t *testing.T) {
	
	// Should be all -1
	a := NewFill(-1, []int{5,4,3})
	b := a.ScalarAdd(1)

	for i := range b.data {
		if b.data[i] != 0 {
			t.Errorf("ScalarAdd did not add correctly")
		}
	}
	
	// Adding a negative num is our subtraction
	c := a.ScalarAdd(-2)
	for i := range c.data {
		if c.data[i] != -3 {
			t.Errorf("ScalarAdd did not subtract correctly")
		}
	}

	// Original tensor should be the same
	for i := range a.data {
		if a.data[i] == -2 {
			t.Errorf("NdArray A should be unaffected by ScalarAdd!")
		}
	}
}

func TestScalarMul(t *testing.T) {
	a := NewFill(10, []int{5,4,3})
	b := a.ScalarMul(2)
	for i := range b.data {
		if b.data[i] != 20 {
			t.Errorf("ScalarMul did not mul correctly")
		}
	}

	for i := range a.data {
		if a.data[i] != 10 {
			t.Errorf("NdArray A should be unaffected by ScalarMul!")
		}
	}
}

func TestNeg(t *testing.T) {
	a := NewFill(-1, []int{5,4,3})
	b := a.Neg()

	for i := range b.data {
		if b.data[i] != 1 {
			t.Errorf("Neg did not flip the sign correctly")
		}
	}

	for i := range a.data {
		if a.data[i] != -1 {
			t.Errorf("NdArray A should be unaffected by Neg")
		}
	}
}

func TestReLu(t *testing.T) {
	a := NewFill(-1, []int{2,2})
	b := a.ReLu()

	for i := range b.data {
		if b.data[i] != 0 {
			t.Errorf("ReLu did not change a negative number")
		}
	}

	for i := range a.data {
		if a.data[i] != -1 {
			t.Errorf("NdArray A should be unaffected by ReLu")
		}
	}

	c := NewFill(1, []int{2,2})
	c = c.ReLu()
	for i := range c.data {
		if c.data[i] != 1 {
			t.Errorf("ReLu changed a positive number")
		}
	}
}
//----------- End Unary Ops -----------

//----------- Begin Binary Ops -----------
func TestAdd(t *testing.T) {
	a := NewFill(3, []int{3,3,3})
	b := NewFill(4, []int{3,3,3})

	c := a.Add(b)

	shape := c.Shape()
	size := c.Size()
	ndim := c.Ndim()

	if ndim != 3 { t.Errorf("Ndim incorrect after add operation") }
	if size != 3*3*3 { t.Errorf("Size incorrect after add operation") }

	for i := range shape {
		if shape[i] != 3 {
			t.Errorf("Shape incorrect after add operation")
		}
	}
}

func BenchmarkAdd(b *testing.B) {
	x := Rand([]int{10,1000,1000})
	y := Rand([]int{10,1000,1000})

	for b.Loop() {
		x.Add(y)
	}
}
//----------- End Binary Ops -----------


//----------- Begin Ops -----------
func TestMul(t *testing.T) {
	a := NewFill(3, []int{2,2})
	b := NewFill(4, []int{2,2})
	c := a.MatMul(b)
	for i := range c.data {
		if c.data[i] != 24 {
			t.Errorf("MatMul result has incorrect data")
		}
	}
	if c.size != 4 {
		t.Errorf("MatMul result has incorrect size")
	}
	for i := range c.shape {
		if c.shape[i] != 2 {
			t.Errorf("MatMul result has incorrect shape")
		}
	}
	
	d := NewFill(3, []int{5,3})
	e := NewFill(2, []int{3,2})
	f := d.MatMul(e)
	for i := range f.data {
		if f.data[i] != 18 {
			t.Errorf("MatMul result has incorrect data")
		}
	}
	if f.size != 5*2 {
		t.Errorf("MatMul result has incorrect size")
	}
	if f.shape[0] != 5 || f.shape[1] != 2 {
		t.Errorf("MatMul doesn't have the correct shape")
	}
	
	x := NewFill(-1.3, []int{3,6,2})
	y := NewFill(5.4, []int{3,2,7})
	z := x.MatMul(y)
	for i := range z.data {
		if z.data[i] != -14.04 {
			t.Errorf("MatMul result has incorrect data")
		}
	}
	if z.size != 3*6*7 {
		t.Errorf("MatMul result has incorrect size")
	}
	if z.shape[0] != 3 || z.shape[1] != 6 || z.shape[2] != 7 {
		t.Errorf("MatMul doesn't have the correct shape")
	}
}

func BenchmarkMatMul(b *testing.B) {
	x := Rand([]int{1,1000,1000})
	y := Rand([]int{1,1000,1000})

	for b.Loop() {
		x.MatMul(y)
	}
}
//----------- End Ops -----------
