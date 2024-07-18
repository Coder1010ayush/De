# ------------ utf-8 encoding ----------------
from autodiff.diff import Tensor
from initializes.random_init import Initializer
from nn.networkx import Linear, Embedding, RNNCell, GRUCell, LSTMCell
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


def testing_stack_operation():

    t1 = Tensor(data=np.random.randn(2, 3), requires_grad=True, dtype=np.float32)
    t2 = Tensor(data=np.random.randn(2, 3), requires_grad=True, dtype=np.float32)
    t3 = Tensor(data=np.random.randn(2, 3), requires_grad=True, dtype=np.float32)
    stack_op = Tensor.stack(0, [t1, t2, t3])

    stack_op.backpropogate()

    # Check the gradients of the input tensors
    print("t1 gradient:\n", t1.grad)
    print("t2 gradient:\n", t2.grad)
    print("t3 gradient:\n", t3.grad)


def testing_flip_operation():
    t1 = Tensor(data=np.random.randn(2, 3), requires_grad=True, dtype=np.float32)
    t2 = Tensor(data=np.random.randn(2, 3), requires_grad=True, dtype=np.float32)
    o1 = t1.flip(axis=(0, 1))
    out = o1 + t2
    f_out = out.mean()
    f_out.backpropogate()

    print("f_out grad is ", f_out.grad)
    print("out grad is ", out.grad)
    print("o1 grad is ", o1.grad)
    print("t2 grad is ", t2.grad)
    print("t1 grad is ", t1.grad)


def testing_embedding_layer():
    data = np.random.randint(low=10, high=2000, size=(1000))
    model = Embedding(vocab_size=1000, dim=768)
    out = model(10)
    out.backpropogate()
    print("output grad is ", out.grad.shape)


def testin_rnn_cell():
    data = Initializer().rand(shape=(6, 500, 10), dtype=np.float32, mean=0.5, std=1.0, requires_grad=True)
    model = RNNCell(input_size=10, hidden_size=20, bias_option=True, non_linear_act="sigmoid")
    hx = Initializer().rand(shape=(500, 20), dtype=np.float32, mean=0.5, std=1.0, requires_grad=True)
    for idx in range(6):
        hx = model(Tensor(data=data.data[idx], requires_grad=True, dtype=np.float32), hx)
        hx.backpropogate()
    print(hx)
    print("output shape is ", hx.shape())


def testing_gru_cell():
    data = Initializer().rand(shape=(6, 500, 10), dtype=np.float32, mean=0.5, std=1.0, requires_grad=True)
    model = GRUCell(input_size=10, hidden_size=20, bias_option=True)
    hx = Initializer().rand(shape=(500, 20), dtype=np.float32, mean=0.5, std=1.0, requires_grad=True)
    for idx in range(6):
        hx = model(Tensor(data=data.data[idx], requires_grad=True, dtype=np.float32), hx)
        hx.backpropogate()
    print(hx)
    print("output shape is ", hx.shape())


def testing_lstm_cell():
    data = Initializer().rand(shape=(6, 20), dtype=np.float32, mean=0.5, std=1.0, requires_grad=True)
    h_prev = Initializer().rand(shape=(6, 40), dtype=np.float32, mean=0.5, std=1, requires_grad=True)
    c_prev = Initializer().rand(shape=(6, 40), dtype=np.float32, mean=0.5, std=1, requires_grad=True)
    model = LSTMCell(input_size=20, hidden_size=40, bias_option=True)
    hx = [h_prev, c_prev]
    hx = model(data, hx)
    hx.backpropogate()
    print(hx)
    print("output shape is ", hx.shape())


if __name__ == "__main__":
    testing_lstm_cell()
