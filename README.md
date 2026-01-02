# minigrad
A minimal implementation of autograd and neural networks. Inspired by [micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch). This library is meant to be educational and not meant for production code.

## Install and dev work

Install:
`go get github.com/NoahSchiro/minigrad@latest`

Import in your code:
```
// You may only need one of these depending on your work
import "github.com/NoahSchiro/minigrad/tensor"
import "github.com/NoahSchiro/minigrad/ndarray"
import "github.com/NoahSchiro/minigrad/nn"
```

### Dev work

Install:
`git clone git@github.com:NoahSchiro/minigrad.git`

Create cuda library following instructions in "Cuda Development"

Run tests: 
`go test ./...`

Benchmark code: 
`go test ./... -bench=.`

### CUDA development

In order to compile the CUDA code, you need:
- NVIDIA GPU
- The appropriate NVIDIA drivers
- The CUDA toolkit

Verify with the following commands:
```bash
# Confirm drivers exist
nvidia-smi 

# Confirm nvcc exists
nvcc --version
```

To compile and install the cuda shared library into your env:
```bash
cd ./pkg/ndarray/cuda/
make install
```

This creates `libcuda.so`, moves it to `/usr/local/lib/` and installs it system wide with `ldconfig`.

## Example

There is an example for the xor problem which can be found in `cmd/xor/`

There is an example for finding the boundary of a circle in `cmd/circle/`

## Features and roadmap

- [x] Autograd engine
- [x] Linear layers
- [ ] RNN, LSTM (targeting v2.1)
- [ ] CNN (targeting v2.2)
- [x] Sigmoid
- [x] ReLU
- [x] SoftMax (targeting v2.1)
- [x] SGD optimizer
- [x] Adam optimizer (targeting v2.1)
- [x] CPU parallelism (targeting v2.1)
- [ ] CUDA support (targeting v2.1)

## SCC Report

The goal is to keep the library under 1k lines of code. This excludes test files and examples.

Didn't really take us long to blow past this goal (57 days since v2.0 was released).

When we add CUDA and C, the goal will be to keep each of those under 500 lines of code combined.

| Language | Files | Lines | Blanks | Comments | Code |
|----------|-------|-------|--------|----------|------|
| Go       | 12    | 1579  | 262    | 137      | 1180 |
