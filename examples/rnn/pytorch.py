from io import open
import random
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn

# Some hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letterToIndex(letter)] = 1
    return tensor.to(DEVICE)

# Turn a line into a <line_length x 1 x N_LETTERS>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor.to(DEVICE)

# Given a softmax output of what category
# it is, convert this to the string version of the category
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
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
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long).to(DEVICE)
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

# Define the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(DEVICE)

rnn       = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES).to(DEVICE)
loss_func = nn.NLLLoss()
optim     = torch.optim.SGD(rnn.parameters(), lr=LR)

# Training for a single example / line
def train(category_tensor, line_tensor):

    # Init a zero filled hidden tensor
    hidden = rnn.initHidden()

    rnn.zero_grad()
    output = None

    # Run through each character of the line
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = loss_func(output, category_tensor)
    loss.backward()

    optim.step()

    return output, loss.item()

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

n_iters     = 100000
print_every = 5000
plot_every  = 1000

# Keep track of losses for plotting
current_loss = 0

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% %.4f %s / %s %s' % (iter, iter / n_iters * 100, loss, line, guess, correct))


# Go through a bunch of examples and record which are correctly guessed
correct = 0
total   = 0
for i in range(10000 + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)

    total += 1
    if guess_i == category_i:
        correct += 1

    if i % 1000 == 0:
        print(f"Accuracy: {(correct / total) * 100}")
        print(f"Line: {line} -> {guess}")
