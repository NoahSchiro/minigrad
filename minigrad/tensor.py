from typing import Callable, Sequence 
import copy

from .autograd import Scalar

class Tensor():

    # TODO: How do I type hint a list of arbitrary dimension?
    def __init__(self, data):
        self.shape = self.get_shape(data) #TODO

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

    def get_shape(self, lst, shape=()):

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

    def __str__(self):
        as_list = self.unary(self.data, lambda x: x.data)
        return f"Tensor({as_list})"

    def __add__(self):
        pass

    def __mul__(self):
        pass
