from minigrad.tensor import Tensor
from minigrad.autograd import Scalar

def test_tensor_eq():

    t1 = Tensor([3])
    t2 = Tensor([3])

    assert t1 == t2

    t1 = Tensor([1])
    t2 = Tensor([3])

    assert t1 != t2

    t1 = Tensor([1, 3])
    t2 = Tensor([1, 3])

    assert t1 == t2

    t1 = Tensor([[1, 3]])
    t2 = Tensor([1, 3])

    assert t1 != t2


def test_tensor_addition():
    t1 = Tensor([[0,1,2], [3,4,5], [6,7,8]])
    t2 = Tensor([[8,7,6], [5,4,3], [2,1,0]])
    t3 = t1+t2

    for i in range(0,3):
        for j in range(0,3):
            assert t3[i][j].data == 8


    t1 = Tensor([[1,1]])
    t2 = Tensor([[4,5,6]])

    try:
        t3 = t1+t2
    except:
        print("exception!")

def test_tensor_mul():
    
    t1 = Tensor([[1,2], [3,4]])
    t2 = Tensor([[5,6], [7,8]])
    t3 = t1*t2
    expected = Tensor([[19, 22], [43, 50]])

    assert t3 == expected

    t1 = Tensor([[1,2], [3,4]])
    t2 = Tensor([[1,0], [0,1]]) # Identity
    t3 = t1*t2
    expected = Tensor([[1, 2], [3, 4]])

    assert t3 == expected

