from .tensor import Tensor
from .autograd import Scalar
from random import uniform
from functools import reduce

class Module():
    def __init__(self):
        raise NotImplementedError()

    def forward(self, data: Tensor):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def __call__(self, data):
        return self.forward(data)

class Linear(Module):
    def __init__(self, in_feat: int, out_feat: int):

        rand = lambda: uniform(0,2) - 1

        self.weights = Tensor([[rand() for _ in range(in_feat)] for _ in range(out_feat)])
        self.biases = Tensor([[rand()] for _ in range(out_feat)])

    def forward(self, data: Tensor):
        return (self.weights * data) + self.biases

    def parameters(self):
        return [self.weights, self.biases]

# Mean squared error
def mse(lhs: Tensor, rhs: Tensor):
    
    assert lhs.shape == rhs.shape

    num_elems = Scalar(reduce(lambda x, y: x*y, lhs.shape))

    def recurse(lhs, rhs, loss):
        for idx in range(0, len(lhs)):
            if isinstance(lhs[idx], list):
                return recurse(lhs[idx], rhs[idx], loss)
            else:
                loss += (rhs[idx] - lhs[idx])**2

        return loss
    
    loss = recurse(lhs.data, rhs.data, Scalar(0))

    return loss / num_elems
