package nn

import t "github.com/NoahSchiro/minigrad/pkg/tensor"

type SGD struct{
	params []*t.Tensor
	lr float32
}

func SGDNew(p []*t.Tensor, lr float32) *SGD {
	return &SGD{
		params: p,
		lr: lr,
	}
}

func (a *SGD) ZeroGrad() {
	for i := range a.params {
		a.params[i].ZeroGrad()
	}
}

func (a *SGD) Update() {
	for i := range a.params {
		t := a.params[i]
		
		for j := range t.Size() {
			new_value := (t.GetLinearGrad(j) * a.lr) + t.GetLinear(j)
			t.SetLinear(j, new_value)
		}
	}
}
