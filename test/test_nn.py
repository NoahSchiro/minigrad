from minigrad.nn import Linear
from minigrad.tensor import Tensor

def test_linear_init():

    l1 = Linear(4,5)

    assert l1.weights.shape == (5,4)
    assert l1.biases.shape  == (5,1)

    assert len(l1.parameters()) == 2

def test_linear_forward():

    t1 = Tensor([[3,4,5]])
    l1 = Linear(3,1)

    t1.transpose()

    result = l1(t1)
