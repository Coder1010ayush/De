# ----------------------- utf-8 encoding --------------------------
from autodiff.diff import Tensor
import numpy as np
import os
import sys

# all the activation function's forward and backward function will be implemented in this file.


class Sigmoid:
    """
        f(x) = 1/ (1+e{-x})  // {} represents the power
        f'(x) = f(x) * (1-f(x))
    """

    def forward(self, inp_tensor: Tensor):
        out = 1/(1+np.exp(-inp_tensor.data))
        return Tensor(data=out, dtype=out.dtype, requires_grad=inp_tensor.requires_grad, inputs_node=[inp_tensor], operation="Backward<Sigmoid>")

    def backward(self, output_node: Tensor):
        input_node = output_node.inputs_node[0]
        input_node.grad = (output_node.data*(1-output_node.data)) * output_node.grad


class Tanh:
    """
        f(x) = e{x} - e{-x} / (e{x} + e{-x})
        f'(x) = 1- f(x)**2
    """

    def forward(self, inp_tensor: Tensor):
        a = np.exp(inp_tensor.data)
        b = np.exp(-inp_tensor.data)
        out = (a-b)/(a+b)
        return Tensor(data=out, requires_grad=inp_tensor.requires_grad, dtype=out.dtype, inputs_node=[inp_tensor], operation="Backward<Tanh>")

    def backward(self, output_node: Tensor):
        input_node = output_node.inputs_node[0]
        input_node.grad = (1-(output_node.data*output_node.data)) * output_node.grad


class Relu:
    """
        f(x) = max(a ,0)
        f'(x) = {
                    1 , a>0
                    0 , otherwise
                }
    """

    def forward(self, inp_tensor: Tensor):
        out = np.maximum(inp_tensor.data, 0)
        return Tensor(data=out, requires_grad=inp_tensor.requires_grad, dtype=out.dtype, inputs_node=[inp_tensor], operation="Backward<Relu>")

    def backward(self, output_node: Tensor):
        input_node = output_node.inputs_node[0]
        input_node.grad = np.where(output_node.data > 0, 1, 0)
