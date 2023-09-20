from __future__ import annotations
from typing import Union, Tuple, Optional

class Scalar():

    def __init__(self,
             data: Union[float, int],
             parents: Tuple[Optional[Scalar], Optional[Scalar]] = (None, None), 
             ):

        self.data = data
        self.grad = 0.0
        self.parents = parents

    # Stringify for debugging
    def __str__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"

    # Addition operation override
    def __add__(self, rhs: Scalar):
        new = Scalar(self.data + rhs.data, parents=(self,rhs))
        return new

    # Subtraction operation override
    def __sub__(self, rhs: Scalar):
        new = Scalar(self.data - rhs.data, parents=(self,rhs))
        return new

    # Multiplication operation override
    def __mul__(self, rhs: Scalar):
        new = Scalar(self.data * rhs.data, parents=(self,rhs))
        return new

    # Division operation override
    def __truediv__(self, rhs: Scalar):
        assert rhs.data != 0, "Division by 0 error in Scalar operation"

        new = Scalar(self.data / rhs.data, parents=(self,rhs))
        return new

    # Power operation, but only support integer / float powers
    def __pow__(self, power: Union[int, float]):
        new = Scalar(self.data**power, parents=(self, None))
        return new

    # Unary operations
    def relu(self):
        new = Scalar(max(self.data, 0), parents=(self, None))
        return new
