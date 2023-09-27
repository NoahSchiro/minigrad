from minigrad.tensor import Tensor
from minigrad.nn import SGD


def test_tensor_autograd():
    t1 = Tensor([[10, 20], [30, 40]])
    t2 = Tensor([[1, 2.1], [3,4]])

    optim = SGD([t1, t2], 0.1)

    for epoch in range(0,2):
        t3 = t1 * t2
        #print(t3)

        t3.data[0][0].backward()

        for i in range(len(t1.data)):
            for j in range(len(t1.data)):
                #print(t2[i][j])
                pass

        optim.step()





test_tensor_autograd()
