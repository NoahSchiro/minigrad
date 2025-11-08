package tensor;

import "testing"
import "math"

//----------- Begin Init functions -----------
func TestTensorNew(t *testing.T) {
	
	a := New([]float32{1}, []int{1})


	if a.data.Shape()[0] != 1 { t.Errorf("Shape of Tensor in New didn't work")}
	if a.data.Size() != 1 { t.Errorf("Size was not computed correctly in New") }
	if a.data.Ndim() != 1 { t.Errorf("Number of dims in New is wrong") }

	elem, err := a.data.Get([]int{0})
	if err != nil {
		t.Errorf("Error in fetching first elem of tensor")
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
		t.Errorf("Error in fetching (1,0) elem of tensor")
	}
	if elem != 0 { t.Errorf("Fetching data from Tensor NdArray was wrong") }
	
	elem, err = a.data.Get([]int{1,1})
	if err != nil {
		t.Errorf("Error in fetching (1,1) elem of tensor")
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
		t.Errorf("Topological ordering is wrong")
	}
	if topo[1] != &b {
		t.Errorf("Topological ordering is wrong")
	}
	if topo[2] != &c {
		t.Errorf("Topological ordering is wrong")
	}
	if topo[3] != &d {
		t.Errorf("Topological ordering is wrong")
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
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != 4 {
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
}

// Test the add with the broadcast function
func TestBroadcastAdd(t *testing.T) {
	a := New([]float32{
		1, 2,
		3, 4,
	}, []int{2, 2})
	b := New([]float32{10, 20}, []int{2})

	c := a.Add(&b)
	c.Backward()

	// Check forward
	expected := []float32{
		11, 22,
		13, 24,
	}
	for i, v := range expected {
		elem, _ := c.data.Get([]int{i / 2, i % 2})
		if elem != v {
			t.Errorf("BroadcastAdd failed at %d: got %v, want %v", i, elem, v)
		}
	}

	// Gradients for a should all be 1
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			aGrad, _ := a.grad.Get([]int{i, j})
			if aGrad != 1 {
				t.Errorf("BroadcastAdd a.grad incorrect at %d,%d: got %v", i, j, aGrad)
			}
		}
	}

	// Gradients for b should be summed over rows
	expectedBGrad := []float32{2, 2}
	for j, v := range expectedBGrad {
		bGrad, _ := b.grad.Get([]int{j})
		if bGrad != v {
			t.Errorf("BroadcastAdd b.grad incorrect at %d: got %v, want %v", j, bGrad, v)
		}
	}
}

func TestScalarMul(t *testing.T) {
	a := New([]float32{1,2}, []int{2})
	b := a.ScalarMul(2)

	b.Backward()

	elem, _ := b.data.Get([]int{0})
	if elem != 2 {
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != 4 {
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != 2 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != 2 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
}

func TestNeg(t *testing.T) {
	a := New([]float32{1,2}, []int{2})
	b := a.Neg()

	b.Backward()

	elem, _ := b.data.Get([]int{0})
	if elem != -1 {
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.data.Get([]int{1})
	if elem != -2 {
		t.Errorf("ScalarAdd did not add correctly")
	}
	elem, _ = b.grad.Get([]int{0})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1})
	if elem != 1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0})
	if elem != -1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{1})
	if elem != -1 {
		t.Errorf("ScalarAdd gradient incorrect")
	}
}

func TestSigmoid(t *testing.T) {
    // Input tensor
    a := New([]float32{-1, 0, 1}, []int{3})
    b := a.Sigmoid()

    // Run backward to compute gradients
    b.Backward()

    // ---- Forward checks ----
    // Sigmoid(x) = 1 / (1 + e^(-x))
    expected := []float32{
        1 / (1 + float32(math.Exp(1))), // sigmoid(-1)
        0.5,                            // sigmoid(0)
        1 / (1 + float32(math.Exp(-1))), // sigmoid(1)
    }

    for i := range expected {
        elem, _ := b.data.Get([]int{i})
        if math.Abs(float64(elem-expected[i])) > 1e-5 {
            t.Errorf("Sigmoid forward incorrect at index %d: got %v, want %v", i, elem, expected[i])
        }
    }

    // ---- Gradient checks ----
    // For sigmoid, dy/dx = sigmoid(x) * (1 - sigmoid(x))
    for i := range expected {
        s := expected[i]
        expectedGrad := s * (1 - s)

        elem, _ := a.grad.Get([]int{i})
        if math.Abs(float64(elem-expectedGrad)) > 1e-5 {
            t.Errorf("Sigmoid gradient incorrect at index %d: got %v, want %v", i, elem, expectedGrad)
        }
    }
}

func TestReLU(t *testing.T) {
    // Input tensor with positive and negative values
    a := New([]float32{-1, 0, 2}, []int{3})
    b := a.ReLu()

    // Run backward to compute gradients
    b.Backward()

    // ---- Forward checks ----
    elem, _ := b.data.Get([]int{0})
    if elem != 0 {
        t.Errorf("ReLU forward incorrect at index 0: got %v, want %v", elem, 0)
    }

    elem, _ = b.data.Get([]int{1})
    if elem != 0 {
        t.Errorf("ReLU forward incorrect at index 1: got %v, want %v", elem, 0)
    }

    elem, _ = b.data.Get([]int{2})
    if elem != 2 {
        t.Errorf("ReLU forward incorrect at index 2: got %v, want %v", elem, 2)
    }

    // ---- Gradient checks ----
    // b.grad should be 1 everywhere after backward (if Backward initializes it that way)
    elem, _ = b.grad.Get([]int{0})
    if elem != 1 {
        t.Errorf("ReLU output gradient incorrect at index 0: got %v, want %v", elem, 1)
    }

    elem, _ = b.grad.Get([]int{1})
    if elem != 1 {
        t.Errorf("ReLU output gradient incorrect at index 1: got %v, want %v", elem, 1)
    }

    elem, _ = b.grad.Get([]int{2})
    if elem != 1 {
        t.Errorf("ReLU output gradient incorrect at index 2: got %v, want %v", elem, 1)
    }

    // ---- Input gradient checks (a.grad) ----
    // d/dx relu(x) = 0 for x <= 0, 1 for x > 0
    elem, _ = a.grad.Get([]int{0})
    if elem != 0 {
        t.Errorf("ReLU input gradient incorrect at index 0: got %v, want %v", elem, 0)
    }

    elem, _ = a.grad.Get([]int{1})
    if elem != 0 {
        t.Errorf("ReLU input gradient incorrect at index 1: got %v, want %v", elem, 0)
    }

    elem, _ = a.grad.Get([]int{2})
    if elem != 1 {
        t.Errorf("ReLU input gradient incorrect at index 2: got %v, want %v", elem, 1)
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
		t.Errorf("Add did not add correctly")
	}
	elem, _ = c.data.Get([]int{1,0})
	if elem != 1 {
		t.Errorf("Add did not add correctly")
	}

	elem, _ = c.grad.Get([]int{1,1})
	if elem != 1 {
		t.Errorf("Add gradient incorrect")
	}
	elem, _ = b.grad.Get([]int{1,0})
	if elem != 1 {
		t.Errorf("Add gradient incorrect")
	}
	elem, _ = a.grad.Get([]int{0,1})
	if elem != 1 {
		t.Errorf("Add gradient incorrect")
	}
}

func TestMatMul(t *testing.T) {

	a := New([]float32{1,2,3}, []int{1,3})
	b := New([]float32{3,3,3}, []int{3,1})
	c := a.MatMul(&b)
	c.Backward()
	
	elem, _ := c.data.Get([]int{0,0})
	if elem != 18 {
		t.Errorf("MatMul incorrect")
	}
	elem, _ = a.grad.Get([]int{0,2})
	if elem != 3 {
		t.Errorf("Gradient isn't right for tensor A")
	}
	elem, _ = b.grad.Get([]int{1,0})
	if elem != 2 {
		t.Errorf("Gradient isn't right for tensor B")
	}
}
//----------- End binary functions -----------
