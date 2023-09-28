from __future__ import annotations
from typing import Callable, Sequence, Tuple
import copy

from .autograd import Scalar

class Tensor():

    # TODO: How do I type hint a list of arbitrary dimension?
    def __init__(self, data) -> None:
        self.shape = self.get_shape(data)

        self.data_list = data
        self.data = None 

        # Make tensor carry Scalar class 
        self.unary(data, lambda x: Scalar(x), True)

    def unary(self, data, func: Callable, reflexive: bool = False):
        acc = copy.deepcopy(data) # Create copy of data

        def recurse(data, acc, func):
            for idx in range(0, len(data)):
                if isinstance(data[idx], list):
                    recurse(data[idx], acc[idx], func)
                else:
                    acc[idx] = func(data[idx])
        
        recurse(data, acc, func) # Write over item

        # Reflexive? then modify own data
        if reflexive:
            self.data = acc
        else:
            return acc # else create a new bunch of data

    def get_shape(self, lst, shape=()) -> Tuple[int]:

        # Base case
        if not isinstance(lst, Sequence):
            return shape

        # Peek ahead and assure all lists in the next depth
        # have the same length
        if isinstance(lst[0], Sequence):
            l = len(lst[0])
            if not all(len(item) == l for item in lst):
                raise ValueError("not all lists have the same length")

        shape += (len(lst), )
        
        # Recurse
        shape = self.get_shape(lst[0], shape)

        return shape

    def zerograd(self):
        def recurse(data):
            for idx in range(0, len(data)):
                if isinstance(data[idx], list):
                    recurse(data[idx])
                else:
                    data[idx].grad = 0.0
        
        recurse(self.data)



    # Just supporting matrix transpose for now
    def transpose(self):
        assert len(self.shape) == 2

        new = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]

        # Transpose
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                new[j][i] = self.data[i][j]

        # Reassign
        self.data = new
        self.shape = [self.shape[1], self.shape[0]]


    def __str__(self) -> str:
        as_list = self.unary(self.data, lambda x: x.data)
        return f"Tensor({as_list})"

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __setitem__(self, idx: int, value: float) -> None:
        self.data[idx] = value

    def __eq__(self, rhs: Tensor) -> bool:

        # If they don't have the same shape then they can't be equal
        if self.shape != rhs.shape:
            return False

        def recurse(lhs, rhs):
            for idx in range(0, len(lhs)):
                if isinstance(lhs[idx], list):
                    recurse(lhs[idx], rhs[idx])
                else:
                    if lhs[idx].data != rhs[idx].data:
                        return False
            return True

        return recurse(self.data, rhs.data)

    def __add__(self, rhs: Tensor) -> Tensor:
        assert self.shape == rhs.shape, "Tensors do not have the same size"
        new = Tensor.zero(self.shape)

        def recurse(lhs, rhs, acc):
            for idx in range(0, len(lhs)):
                if isinstance(lhs[idx], list):
                    recurse(lhs[idx], rhs[idx], acc[idx])
                else:
                    acc[idx] = lhs[idx] + rhs[idx]
        recurse(self.data, rhs.data, new.data)

        return new 

    def __mul__(self, rhs: Tensor) -> Tensor:
        assert len(self.shape) == 2, "Tensor must be matrix (dim=2) for multiplication"
        assert len(rhs.shape)  == 2, "Tensor must be matrix (dim=2) for multiplication"

        assert self.shape[1] == rhs.shape[0], "Tensors cannot be multiplied"

        m = self.shape[0]
        n = self.shape[1]
        p = rhs.shape[1]

        result = Tensor.zero([m, p])

        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += self[i][k] * rhs[k][j]

        return result

    def relu(self) -> Tensor:
        new = Tensor.zero(self.shape)

        def recurse(data, new):
            for idx in range(0, len(data)):
                if isinstance(data[idx], list):
                    recurse(data[idx], new[idx])
                else:
                    new[idx] = data[idx].relu()
        recurse(self.data, new.data)

        return new

    def sigmoid(self) -> Tensor:
        new = Tensor.zero(self.shape)

        def recurse(data, new):
            for idx in range(0, len(data)):
                if isinstance(data[idx], list):
                    recurse(data[idx], new[idx])
                else:
                    new[idx] = data[idx].sigmoid()
        recurse(self.data, new.data)

        return new

    @staticmethod
    def zero(shape: list[int]) -> Tensor:
        new = []
        for _ in range(shape[0]):
            temp = [0] * shape[1]
            new.append(temp)

        return Tensor(new)
        
