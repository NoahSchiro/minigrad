# minigrad
A minimal implementation of neural networks in Python. Inspired by [micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch)

-------------

Use:
Similar to pytorch, this is really a tensor library with a thin neural networks wrapper built on top. The following shows tensor intialization and a few of the operations available
```
from minigrad.tensor import tensor

a = Tensor([[1, 2], [3, 4]])
b = Tensor([[1, 2], [3, 4]])

c = a+b
d = a*b

d_relu = d.relu() # Element-wise relu
c_sig  = c.sigmoid() # Element-wise sigmoid
```

Naturally, the most important aspect of a machine learning library is autogradient. Similar to the pytorch api, we can just call backward on any tensor to compute gradients

```
c_sig.backward()
```

Finally, if we wanted to actually step our tensor parameters in the direction that minimizes `c_sig`, then we just have to hand off the parameters to a stochastic gradient decent engine

```
sgd = SGD([a, b], lr=0.01)

# Once gradients are computed, just call step!
sgd.step()
```

To create your own neural networks, you need to extend off of nn.Module, again a very similar idea to pytorch. When you create your own module, there are three functions you need to define
```
from minigrad import nn

def MyNet(nn.Module):
    def __init__(self):
        # Stateful information goes here!
        pass

    def forward(self, x):
        # What do we do when we want to pass an input through the network?
        return tensor

    def parameters():
        # Here we return all stateful information as a list
        return 
```

There is much more functionality, but to learn more, check out the examples directory!
