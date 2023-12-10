from io import open
import random
import glob
import os
import unicodedata
import string

import sys
sys.path.append("../../")
import minigrad.nn as nn
from minigrad.tensor import Tensor
from minigrad.autograd import Scalar 

# Some hyperparameters
LR = 0.005
N_HIDDEN = 128


# List of all possible letters and the length of this list
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Turn a Unicode string to plain ASCII, thanks
# to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Find letter index from ALL_LETTERS, e.g. "a" = 0
def letterToIndex(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x N_LETTERS> Tensor
def letterToTensor(letter):
    tensor = Tensor.zero([1, N_LETTERS])
    tensor[0][letterToIndex(letter)].data = 1.
    return tensor

# Turn a line into a <line_length x 1 x N_LETTERS>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensors = []

    for letter in line:
        tensors.append(letterToTensor(letter))

    return tensors

# Given a softmax output of what category
# it is, convert this to the string version of the category
def categoryFromOutput(output: Tensor):
    top_n, top_i = output.topk(1) # TODO: Need a topk
    category_i = top_i[0].item() # TODO: we don't have an "item"
    return all_categories[category_i], category_i

# Given a list, select a random element
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Select a random training example
def randomTrainingExample():
    # Random category and line within that category
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])

    # Convert the category and line to tensor versions
    category_tensor = Torch.tensor([all_categories.index(category)])
    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

N_CATEGORIES = len(all_categories)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def initHidden(self):
        return Tensor.zero([1, self.hidden_size]) 

    def forward(self, input): 
        input.transpose()

        print(combined.shape)
        print(self.i2h.weights.shape)

        # Pass it through the first set of linear stuff
        hidden = self.i2h(combined)

        # After we have gotten the new hidden vector we can also get the output
        output = self.h2o(hidden)

        # Need to implement a softmax!
        #output = softmax(output)
        return output, hidden

    def parameters(self):
        return self.i2h.parameters() + self.h2o.parameters()


model = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES)

hidden_vec = model.initHidden()
input_tensor = letterToTensor("A")

# First we need to combine the tensors
combined = Tensor([input_tensor.data[0] + hidden_vec.data[0]])

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Hidden tensor shape: {hidden_vec.shape}")
print(f"Combined shape: {combined.shape}")


out, hid = model(combined)
print(out.shape)
print(hid.shape)

