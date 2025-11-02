package tensor;

func checkShape(a Tensor, b Tensor) bool {

	a_shape := a.data.Shape()
	b_shape := b.data.Shape()

	if len(a_shape) != len(b_shape) {
		return false
	}

	for i := range a_shape {
		if a_shape[i] != b_shape[i] {
			return false
		}
	}

	return true
}

