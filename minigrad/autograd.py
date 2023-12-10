from __future__ import annotations
from typing import Union, Tuple, Optional
from math import exp

class Scalar():

    def __init__(self,
             data: Union[float, int, Scalar],
             parents: Tuple[Optional[Scalar], Optional[Scalar]] = (None, None), 
             op: str = ''):

        if isinstance(data, Scalar):
            self.data = data.data
        else:
            self.data = data          # The actual data being stored

        self.grad = 0.0               # The gradient of this value
        self.parents = set(parents)   # Where this value came from 
        self._backward = lambda: None # The function to be called on backprop
        self._op = op                 # The operation this value came from

    def backward(self):
        # Compute a topological ordering of the tree that created this scalar
        t = []
        visisted = set()
        def build_t(s: Scalar):
            if s not in visisted:
                visisted.add(s)
                for parent in s.parents:
                    if parent: build_t(parent)
                t.append(s)
        build_t(self)

        # Compute gradients in backwards order of topological ordering
        self.grad = 1 # Naturally the derivative of x with respect to itself is 1
        for s in reversed(t): s._backward()

    # Stringify for debugging
    def __str__(self):
        return f"Scalar(data={self.data}, grad={self.grad} op={self._op})"

    # Addition operation override
    def __add__(self, rhs: Scalar):
        new = Scalar(self.data + rhs.data, parents=(self,rhs), op='+')

        # Define the derivative calculation for this operation
        def backward():
            self.grad += new.grad
            rhs.grad += new.grad
        new._backward = backward

        return new

    # Subtraction operation override
    def __sub__(self, rhs: Scalar):
        return self + (-rhs)

    # Multiplication operation override
    def __mul__(self, rhs: Scalar):
        new = Scalar(self.data * rhs.data, parents=(self,rhs), op='*')

        # Define the derivative calculation for this operation
        def backward():
            self.grad += rhs.data * new.grad
            rhs.grad += self.data * new.grad
        new._backward = backward

        return new

    # Division operation override
    def __truediv__(self, rhs: Scalar):
        return self * (rhs**-1)

    # Power operation, but only support integer / float powers
    def __pow__(self, power: Union[int, float]):
        new = Scalar(self.data**power, parents=(self, None), op='pow')

        # Define the derivative calculation for this operation
        def backward():
            self.grad += (power * self.data**(power-1)) * new.grad
        new._backward = backward

        return new

    # Unary operations

    # Negate operation
    def __neg__(self): # -self
        return self * Scalar(-1)

    # ReLU
    def relu(self):
        new = Scalar(max(self.data, 0), parents=(self, None), op='relu')

        # Define the derivative calculation for this operation
        def backward():
            self.grad += (new.data > 0) * new.grad
        new._backward = backward

        return new

    def sigmoid(self):
        euler = exp(-self.data)
        new = Scalar((1 / (1 + euler)), parents=(self, None), op='sigmoid')

        def backward():
            self.grad += (euler / (euler + 1)**2) * new.grad
        new._backward = backward

        return new
