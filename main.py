from minigrad.tensor import Tensor
from minigrad.nn import Linear


t1 = Tensor([[1,2,3]])
t1.transpose()

# Input dimensions 3, output 2
# Does weight multiplication and biases
lin = Linear(3, 2)

print(f"Tensor: {t1}")

print("Linear model: ")
weights, biases = lin.parameters()

print(f"Weights: {weights}")
print(f"Biases: {biases}")


print(f"Result: {lin(t1)}")
