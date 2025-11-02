# minigrad
A minimal implementation of autograd and neural networks. Inspired by [micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch). This library is meant to be educational and not meant for production code.

## Install and dev work:

Install:
`go get github.com/NoahSchiro/minigrad@latest`

Import in your code:
```
// You may only need one of these depending on your work
import "github.com/NoahSchiro/minigrad/tensor"
import "github.com/NoahSchiro/minigrad/ndarray"
import "github.com/NoahSchiro/minigrad/nn"
```

### Dev work:

Install:
`git clone git@github.com:NoahSchiro/minigrad.git`

Run tests: 
`go test ./...`

## Example:

Currently, there is one example for the xor problem which can be found in `cmd/xor.go`

## Features and roadmap:

- [x] Autograd engine
- [x] Linear layers
- [ ] RNN, LSTM (targeting v2.1)
- [ ] CNN (targeting v2.2)
- [x] Sigmoid
- [ ] ReLU, SoftMax (targeting v2.1)
- [x] SGD optimizer
- [ ] Adam optimizer (targeting v2.1)

## SCC Report

The goal is to keep the library under 1k lines of code. This excludes test files.

| Language | Files | Lines | Blanks | Comments | Code |
|----------|-------|-------|--------|----------|------|
| Go | 12 | 1016 | 163 | 74 | 779 |
