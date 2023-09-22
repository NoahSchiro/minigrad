from minigrad.tensor import Tensor
from minigrad.autograd import Scalar

def test_tensor_addition():
    t1 = Tensor([[0,1,2], [3,4,5], [6,7,8]])
    t2 = Tensor([[8,7,6], [5,4,3], [2,1,0]])
    t3 = t1+t2

    for i in range(0,3):
        for j in range(0,3):
            assert t3.data[i][j].data == 8


    t1 = Tensor([[1,1]])
    t2 = Tensor([[4,5,6]])

    try:
        t3 = t1+t2
    except:
        print("exception!")




