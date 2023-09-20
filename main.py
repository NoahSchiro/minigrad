from minigrad.autograd import Scalar

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

testing_operations()
