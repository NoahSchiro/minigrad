package main 

import (
	"fmt"
	"math"
	"image"
	"image/color"
	"image/png"
	"os"
)
import t "github.com/NoahSchiro/minigrad/pkg/tensor"
import nn "github.com/NoahSchiro/minigrad/pkg/nn"

const LR float32 = 0.01
const SAMPLES int = 5000 
const EPOCHS int = 200

type Model struct {
	l1 *nn.Linear
	l2 *nn.Linear
	l3 *nn.Linear
}
func ModelNew() *Model {
	return &Model{
		l1: nn.LinearNew(2,16),
		l2: nn.LinearNew(16,16),
		l3: nn.LinearNew(16,1),
	}
}
func (a *Model) Forward(input *t.Tensor) *t.Tensor {
	l1_out := a.l1.Forward(input).ReLu()
	l2_out := a.l2.Forward(l1_out).ReLu()
	l3_out := a.l3.Forward(l2_out).Sigmoid()
	return l3_out
}
func (a *Model) Parameters() []*t.Tensor {
	l1p := a.l1.Parameters()
	l2p := a.l2.Parameters()
	l3p := a.l3.Parameters()
	concat := append(l1p, l2p...)
	concat = append(concat, l3p...)
	return concat
}

func test(epoch int, net *Model, testData *t.Tensor) error {
	imgDim := 600
	w, h := imgDim, imgDim 
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	dotSize := 2

	testResult := net.Forward(testData)
	
	for i := range testData.Shape()[0] {
		x := int((testData.Get([]int{i, 0}) + 1) * 0.5 * (float32(imgDim)-1))
		y := int((testData.Get([]int{i, 1}) + 1) * 0.5 * (float32(imgDim)-1))

		pred := testResult.Get([]int{i, 0})

		var c color.RGBA
		if pred >= 0.5 {
			c = color.RGBA{255, 0, 0, 255} // red
		} else {
			c = color.RGBA{0, 0, 255, 255} // blue
		}

		// Draw a filled dot
        for dx := -dotSize; dx <= dotSize; dx++ {
            for dy := -dotSize; dy <= dotSize; dy++ {
                xx := x + dx
                yy := y + dy
                if xx >= 0 && xx < w && yy >= 0 && yy < h {
                    if dx*dx+dy*dy <= dotSize*dotSize {
                        img.Set(xx, yy, c)
                    }
                }
            }
        }
		img.Set(x, h-1-y, c)
	}

	fname := fmt.Sprintf("cmd/circle/imgs/frame_%03d.png", epoch)
	f, err := os.Create(fname)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", fname, err)
	}
	defer f.Close()
	return png.Encode(f, img)
}

func main() {

	// Create data in range [-1, 1]
	x := t.Rand([]int{SAMPLES, 2})
	x = x.ScalarMul(2).ScalarAdd(-1)

	x_test := t.Rand([]int{SAMPLES, 2})
	x_test = x_test.ScalarMul(2).ScalarAdd(-1)

	// Calculate whether data is in the circle
	y_data := make([]float32, SAMPLES)
	for i := range SAMPLES {
		a := float64(x.Get([]int{i, 0}))
		b := float64(x.Get([]int{i, 1}))
		var in_circle float32 = 1.
		if math.Sqrt(math.Pow(a, 2) + math.Pow(b, 2)) > 0.75 {
			in_circle = 0.
		}
		y_data[i] = in_circle
	}

	y := t.New(y_data, []int{SAMPLES, 1})

	model := ModelNew()
	optim := nn.AdamNew(
		model.Parameters(),
		LR,
		0.9, 0.999, 1e-8,
	)

	for i := range EPOCHS {
		pred := model.Forward(x)
		loss := nn.MSE(&y, pred)
		if i % 100 == 0 {
			fmt.Println(loss.Print())
		}
		loss.Backward()
		optim.Update()
		optim.ZeroGrad()

		test(i, model, x_test)
	}
}
