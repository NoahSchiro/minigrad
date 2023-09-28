import sys
sys.path.append("../../")
import minigrad.nn as nn
from minigrad.tensor import Tensor
from minigrad.autograd import Scalar 

class XOR(nn.Module):

    def __init__(self):

        # Input to hidden
        self.l1 = nn.Linear(2,2)

        # Hidden to output
        self.l2 = nn.Linear(2,1)

    def forward(self, x):
        x.transpose()

        # Feed forward
        x = self.l1(x)
        x = x.sigmoid()
        x = self.l2(x)

        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

X_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

Y_data = [
    [0],
    [1],
    [1],
    [0],
]

def train(model, optim):

    acc_loss = Scalar(0)

    # Treat the whole dataset as one batch
    for (X,Y) in zip(X_data, Y_data):
        # Clear gradients
        model.zerograd()

        X = Tensor([X])
        Y = Tensor([Y])

        # Predict
        prediction = model(X)

        # Calculate loss
        loss = nn.mse(Y, prediction)

        acc_loss += loss

        # Backward pass
        loss.backward()

        # Step the parameters once we have done a batch
        optim.step()


def test(model):
    acc_loss = Scalar(0)
    correct = 0
    total   = 0

    # Treat the whole dataset as one batch
    for (X,Y) in zip(X_data, Y_data):
        X = Tensor([X])
        Y = Tensor([Y])

        # Predict
        prediction = model(X)

        # Calculate loss
        loss = nn.mse(Y, prediction)
        acc_loss += loss

        if round(prediction.data[0][0].data) == Y.data[0][0].data:
            correct += 1
        total += 1

    print(f"Loss: {acc_loss.data:.4f}")
    print(f"Correct: {(correct / total) * 100}")

def inference(model):
    for (X,Y) in zip(X_data, Y_data):
        X = Tensor([X])
        Y = Tensor([Y])

        prediction = model(X)

        print(f"{X[0][0].data}^{X[1][0].data} = {prediction[0][0].data:.4f}")


model = XOR()
optim = nn.SGD(model.parameters(), 0.01)

for epoch in range(1, 15001):
    train(model, optim)

    if epoch % 5000 == 0:
        print(f"Epoch: {epoch}")
        test(model)
        inference(model)


