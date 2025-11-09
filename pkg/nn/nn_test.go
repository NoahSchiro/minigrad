package nn

import "testing"
import "fmt"
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

	fmt.Println(out2.Print())

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

// func TestMSE(t *testing.T) {
//
// }
//----------- End Loss functions -----------
