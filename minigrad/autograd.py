from typing import Union


class Scalar():

    def __init__(self,
             data: Union[float, int],
             parents = (), # How can I declare a that parents is Tuple[Scalar] here?
             ):

        self.data = data
        self.grad = 0.0
        self.parents = parents

    # Stringify for debugging
    def __str__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"

    # Addition operation override
    def __add__(self, rhs):
        new = Scalar(self.data + rhs.data)
        return new

    # Subtraction operation override
    def __sub__(self, rhs):
        new = Scalar(self.data - rhs.data)
        return new

    # Multiplication operation override
    def __mul__(self, rhs):
        new = Scalar(self.data * rhs.data)
        return new

    # Division operation override
    def __truediv__(self, rhs):
        assert rhs.data != 0, "Division by 0 error in Scalar operation"

        new = Scalar(self.data / rhs.data)
        return new
