import torch
import torch.nn as nn
import torch.optim as optim

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()

        self.l1 = nn.Linear(2,2)
        self.l2 = nn.Linear(2,1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        return x

X_data = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
])

Y_data = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.],
])

def train(model, lf, optim):

    for (X,Y) in zip(X_data, Y_data):
        X = X[None, :]
        Y = Y[None, :]

        optim.zero_grad()
        pred = model(X)
        loss = lf(Y, pred)
        loss.backward()
        optim.step()

def test(model, lf):
    correct = 0
    total = 4

    loss = 0

    for (X,Y) in zip(X_data, Y_data):
        X = X[None, :]
        Y = Y[None, :]

        pred = model(X)
        print(f"{X} = {pred.item()}")

        loss += lf(Y, pred)

        if torch.round(pred) == Y:
            correct += 1

    return f"loss: {loss:.4f}, acc: {(correct/total) * 100}"


model = XOR()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

for epoch in range(1, 15001):
    
    train(model, loss_func, optim)

    if epoch % 5000 == 0:
        print(f"Epoch: {epoch}")
        print(test(model,loss_func))
