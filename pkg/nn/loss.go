package nn

import t "github.com/NoahSchiro/minigrad/pkg/tensor"

func AbsErr(t1 *t.Tensor, t2 *t.Tensor) *t.Tensor {

	if t1.Ndim() != t2.Ndim() {
		panic("StandardErr tensor dimensions must match")
	}
	t1_shape := t1.Shape()
	t2_shape := t2.Shape()
	for i := range t1_shape {
		if t1_shape[i] != t2_shape[i] {
			panic("StandardErr tensor shape must match")
		}
	}

	sub_term := t2.Neg()

	diff := t1.Add(sub_term) // add a neg num
	abs_value := diff.Abs()
	return abs_value.Sum()
}

func MSE(t1 *t.Tensor, t2 *t.Tensor) *t.Tensor {

	if t1.Ndim() != t2.Ndim() {
		panic("MSE tensor dimensions must match")
	}
	t1_shape := t1.Shape()
	t2_shape := t2.Shape()
	for i := range t1_shape {
		if t1_shape[i] != t2_shape[i] {
			panic("MSE tensor shape must match")
		}
	}

	// Compute (t1 - t2)
	t2Neg := t2.Neg()
	diff := t1.Add(t2Neg)

	// Square element-wise
	sq := diff.ScalarPow(2)

	// Compute average
	sum := sq.Sum()
	n := float32(t1.Size())
	return sum.ScalarDiv(n)
}
