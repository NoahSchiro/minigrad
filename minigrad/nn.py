from .tensor import Tensor
from random import uniform

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
