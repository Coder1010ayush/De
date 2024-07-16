# ------------ utf-8 encoding ----------------
from autodiff.diff import Tensor
from initializes.random_init import Initializer
from nn.linear import Linear
from nn.module import Module, Sequential
import os
import sys
import json
import numpy as np


def test1():
    ob1 = Tensor(data=[1, 2, 3], dtype=np.float16, requires_grad=True)
    ob2 = Tensor(data=[3, 10, 12], dtype=np.float16, requires_grad=True)
    out1 = ob1 + ob2
    c = Tensor(data=[3, 6, 9], dtype=np.float16, requires_grad=True)
    out2 = c.log()
    out = out1 + out2
    o = out.mean()
    o.backpropogate()
    print("obj1 is ", ob1)
    print("obk2 is ", ob2)
    print("c is ", c)
    print("out1 is ", out1)
    print("out2 is ", out2)
    print("out is ", out)
    print()
    print("=========================================")
    print()
    print("final output grad is ", o.grad)
    print("third output grad is ", out.grad)
    print("second output grad is ", out2.grad)
    print("first out grad is ", out1.grad)
    print("input1 grad is ", ob1.grad)
    print("input2 grad is ", ob2.grad)
    print("input3 grad is ", c.grad)


def test_second():
    x1 = Initializer().randn(shape=(300, 20, 20), dtype=np.float32, requires_grad=True)
    weight = Initializer().randn(shape=(1, 20), dtype=float, requires_grad=True)
    weight.transpose()
    bias = Tensor(data=0, requires_grad=True, dtype=np.float32)
    out1 = x1.matmul(weight)
    out = out1 + bias

    print("x1 shape is ", x1.shape())
    print("weight shape is ", weight.shape())
    print("bias shape is ", bias.shape())
    print("out1 shape is ", out1.shape())
    print("out shape is ", out.shape())
    out.backpropogate()


def testing_simple_linear_layer():
    model = Linear(in_features=10, out_features=5, bias_option=True)
    data = Tensor(data=np.random.rand(300, 10), requires_grad=True, dtype=np.float64)
    out = model(data)
    print(out)
    print(out.shape())
    out.backpropogate()


def testin_sequential_layer():
    data = Tensor(data=np.random.rand(300, 20), requires_grad=True, dtype=np.float64)
    model = Sequential(
        Linear(in_features=20, out_features=10, bias_option=True),
        Linear(in_features=10, out_features=20, bias_option=True)
    )
    out = model(data)
    out.backpropogate()
    return out


if __name__ == "__main__":
    # testing_simple_linear_layer()
    out = testin_sequential_layer()
    print(out)
    print(out.shape())
