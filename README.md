# minigrad
A minimal implementation of autograd and neural networks. Inspired by [micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch). This library is meant to be educational and not necessarily meant for production code.

This software has two versions.

v1.0 is an implementation in Python and treats floats as atomic values. It is designed to be minimal and clocks in under 500 lines of code.

v2.0 is written in Go for some additional speed, a new challenge for me, and because Go is a simple language. This simplicity lends itself well to explaining the concepts of an autogradient engine. In this version, we will be treating matrices as the atomic value (as does most modern ML libraries) and we aim to come in under 1000 lines of code with more features than v1.0.

## Features:

- [ ] Autograd engine
- [ ] Linear layers
- [ ] RNN, LSTM
- [ ] CNN
- [ ] ReLU, SoftMax, Sigmoid
- [ ] SGD and Adam optimizer
