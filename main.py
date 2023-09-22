from minigrad.autograd import Scalar
from minigrad.tensor import Tensor

def testing_operations():

    a = Scalar(4)
    b = Scalar(4.5)
    print("Original values:")
    print(a)
    print(b)

    add = a+b
    sub = a-b
    mul = a*b
    div = a/b
    pow = a**3
    rel = a.relu()

    print(f"Addition:       {add}")
    print(f"Subtraction:    {sub}")
    print(f"Multiplication: {mul}")
    print(f"Division:       {div}")
    print(f"a**3:           {pow}")
    print(f"a.relu():       {rel}")
    print(f"sub.relu():     {sub.relu()}")

def testing_autograd():

    a = Scalar(-3.9)
    b = Scalar(1.9)
    g = (a+b) / (a*b)
    g.backward()
    amg, bmg, gmg = a, b, g

    print(amg.data)
    print(bmg.data)
    print(gmg.data)
    print(amg.grad)
    print(bmg.grad)
    print(gmg.grad)

def testing_tensors():
    lst = [[0,1,2], [3,4,5], [6,7,8]]
    t1 = Tensor(lst)

    print(t1)


testing_tensors()
