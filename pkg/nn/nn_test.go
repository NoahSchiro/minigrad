package nn

import "testing"
import "math"
import tensor "github.com/NoahSchiro/minigrad/pkg/tensor"

//----------- Begin Loss functions -----------
func TestAbsErr(t *testing.T) {

	a := tensor.New([]float32{1, 2, 3}, []int{3})
	b := tensor.New([]float32{3, 2, 1}, []int{3})

	out := AbsErr(&a, &b)
	out.Backward()
	
	if out.GetLinear(0) != 4 {
		t.Errorf("AbsErr incorrect: expected 4")
	}
	if a.GetLinearGrad(0) != -1 {
		t.Errorf("Gradient mismatch at a.0")
	}
	if a.GetLinearGrad(1) != 0 {
		t.Errorf("Gradient mismatch at a.1")
	}
	if a.GetLinearGrad(2) != 1 {
		t.Errorf("Gradient mismatch at a.2")
	}

	// ---- Identical tensors ----
	a2 := tensor.New([]float32{1, 2, 3}, []int{3})
	b2 := tensor.New([]float32{1, 2, 3}, []int{3})

	out2 := AbsErr(&a2, &b2)
	out2.Backward()

	if out2.GetLinear(0) != 0 {
		t.Errorf("AbsErr for identical tensors should be 0")
	}
	if a2.GetLinearGrad(0) != 0 {
		t.Errorf("Gradient mismatch at a2.0")
	}
	if a2.GetLinearGrad(1) != 0 {
		t.Errorf("Gradient mismatch at a2.1")
	}
	if a2.GetLinearGrad(2) != 0 {
		t.Errorf("Gradient mismatch at a2.2")
	}
}

func TestMSE(t *testing.T) {

	// ---- Basic case ----
	a := tensor.New([]float32{1, 2, 3}, []int{3})
	b := tensor.New([]float32{3, 2, 1}, []int{3})

	out := MSE(&a, &b)
	out.Backward()

	// Expected MSE:
	// ((1-3)^2 + (2-2)^2 + (3-1)^2) / 3
	// = (4 + 0 + 4) / 3
	// = 8/3
	// approx. 2.6667
	if math.Abs(float64(out.GetLinear(0))-8.0/3.0) > 1e-4 {
		t.Errorf("MSE incorrect: expected approx 2.6667, got %v", out.GetLinear(0))
	}

	// Expected gradient wrt a:
	// d/dx ((1/n) * sum(x_i - y_i)^2) = (2/n) * (x_i - y_i)
	// a = [1,2,3], b = [3,2,1] -> diff = [-2,0,2]
	// grad = (2/3)*[-2,0,2] = [-1.3333, 0, 1.3333]

	expectedGrads := []float32{-1.3333, 0.0, 1.3333}
	for i, exp := range expectedGrads {
		got := a.GetLinearGrad(i)
		if math.Abs(float64(got-exp)) > 1e-3 {
			t.Errorf("Gradient mismatch at a.%d: expected %.4f, got %.4f", i, exp, got)
		}
	}

	// ---- Identical tensors ----
	a2 := tensor.New([]float32{1, 2, 3}, []int{3})
	b2 := tensor.New([]float32{1, 2, 3}, []int{3})

	out2 := MSE(&a2, &b2)
	out2.Backward()

	if out2.GetLinear(0) != 0 {
		t.Errorf("MSE for identical tensors should be 0, got %v", out2.GetLinear(0))
	}
	for i := 0; i < 3; i++ {
		if a2.GetLinearGrad(i) != 0 {
			t.Errorf("Gradient mismatch at a2.%d: expected 0, got %v", i, a2.GetLinearGrad(i))
		}
	}
}
//----------- End Loss functions -----------
