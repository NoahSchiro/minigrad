from minigrad.autograd import Scalar

# Test operator functionality
def test_add():
    a = Scalar(3.2)
    b = Scalar(4.3)
    c = a+b

    assert c.data == (3.2+4.3)

def test_sub():
    a = Scalar(3.2)
    b = Scalar(4.3)
    c = a-b

    assert c.data == (3.2-4.3)

def test_mul():
    a = Scalar(3.2)
    b = Scalar(4.3)
    c = a*b

    assert c.data == (3.2*4.3)

def test_div():
    a = Scalar(3)
    b = Scalar(4)
    c = a/b

    assert c.data == (3/4)

    a.data += 9
    c = a/b
    assert c.data == 3

def div_by_zero():
    a = Scalar(100)
    b = Scalar(0)

    try:
        c = a / b
    except:
        raise ZeroDivisionError("Division by zero")

def test_power():
    a = Scalar(3.2)
    b = a**3

    assert b.data == (3.2**3)

def test_relu():
    a = Scalar(3.2)
    b = Scalar(-4.3)

    assert a.relu().data == 3.2
    assert b.relu().data == 0.0

def test_sigmoid():

    a = Scalar(4)
    b = a.sigmoid()
    b.backward()

    assert (b.data - 0.982014) < 0
    assert (b.grad == 1)
    assert (a.grad - 0.17) < 0

import torch
# Testing our autograd algorithm
def test_autograd():

    a = Scalar(-4.0)
    b = Scalar(2.0)
    c = a + b
    d = a * b + b**3
    c += c + Scalar(1)
    c += Scalar(1) + c + (-a)
    d += d * Scalar(2) + (b + a).relu()
    d += Scalar(3) * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / Scalar(2.0)
    g += Scalar(10.0) / f
    
    g.backward()

    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
