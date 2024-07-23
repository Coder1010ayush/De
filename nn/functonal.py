# ----------------------- utf-8 encoding --------------------------
from autodiff.ops import Relu, Sigmoid, Tanh, Softmax
import numpy as np
import os
import sys

# all the activation function's forward and backward function will be implemented in this file.


def sigmoid(inp_tensor):
    return Sigmoid().forward(inp_tensor=inp_tensor)


def relu(inp_tensor):
    return Relu().forward(inp_tensor=inp_tensor)


def tanh(inp_tensor):
    return Tanh().forward(inp_tensor=inp_tensor)


def sofmax(inp_tensor, dim):
    return Softmax().forward(inp_tensor, dim)
