package nn

import t "github.com/NoahSchiro/minigrad/pkg/tensor"
import "math"

// Begin SGD
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
			new_value := t.GetLinear(j) - (t.GetLinearGrad(j) * a.lr)
			t.SetLinear(j, new_value)
		}
	}
}
// End SGD
// Begin Adam
type Adam struct{
	params []*t.Tensor
	lr float32
	b1 float32    // weighting factor for m
	b2 float32    // weighting factor for v
	eps float32   // ensures non-zero denominator
	m []*t.Tensor // mean of gradients
	v []*t.Tensor // mean of squared gradients
	tstep int
}

func AdamNew(params []*t.Tensor, lr, b1, b2, eps float32) *Adam {
	m := make([]*t.Tensor, len(params))
	v := make([]*t.Tensor, len(params))

	for i, p := range params {
		m[i] = t.Zero(p.Shape())
		v[i] = t.Zero(p.Shape())
	}

	return &Adam{
		params: params,
		lr: lr,
		b1: b1,
		b2: b2,
		eps: eps,
		m: m,
		v: v,
		tstep: 0,
	}
}

func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}

func (a *Adam) Update() {
	a.tstep++

    // Precompute bias correction terms once
    biasCorr1 := float32(1 - math.Pow(float64(a.b1), float64(a.tstep)))
    biasCorr2 := float32(1 - math.Pow(float64(a.b2), float64(a.tstep)))

	for i, p := range a.params {

		// Get avgs
		m := a.m[i]
		v := a.v[i]

		for j := range p.Size() {
			g := p.GetLinearGrad(j)

			// Update moment estimates
			mNew := a.b1*m.GetLinear(j) + (1-a.b1)*g
			vNew := a.b2*v.GetLinear(j) + (1-a.b2)*g*g

			m.SetLinear(j, mNew)
			v.SetLinear(j, vNew)

			// Bias correction
			mHat := mNew / biasCorr1
			vHat := vNew / biasCorr2

			// Parameter update
			newVal := p.GetLinear(j) - a.lr*mHat/(float32(math.Sqrt(float64(vHat)))+a.eps)
			p.SetLinear(j, newVal)
		}
	}
}
// End Adam
