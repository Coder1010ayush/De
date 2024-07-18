# ------- utf-8 encoding -------------
import autodiff
import os
import sys
import numpy as np
from autodiff.utility import data_to_numpy
TENSOR_COUNTER = 0
mode = "forward"


class Tensor:

    def __init__(self, data, dtype, requires_grad=False, operation=None, inputs_node=[], axis=None, params=[]) -> None:
        """_summary_

        Args:
            data (_type_): _description_
            dtype (_type_): _description_
            requires_grad (bool, optional): _description_. Defaults to False.
            operation (_type_, optional): _description_. Defaults to None.
            inputs_node (list, optional): _description_. Defaults to [].
        """
        self.data = data_to_numpy(data=data, dtype=dtype)
        self.requires_grad = requires_grad
        self.inputs_node = inputs_node
        self.dtype = dtype
        self.operation = operation
        self.id = id(self)
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        self.grad = None
        self.axis = axis
        self.params = (params)  # keep_dims , dilation

    def shape(self):
        return self.data.shape

    # how tensor will be looking when printing in the console

    def __repr__(self) -> str:
        srepr = f"Tensor({self.data},{self.dtype})"
        return srepr

    def clip_grad(self, min_val=-1e10, max_val=1e10):
        if self.grad is not None:
            np.clip(self.grad, min_val, max_val, out=self.grad)

    # backpropogation function
    def backpropogate(self):
        if not self.requires_grad:
            raise ValueError("Gradient tracking is not enabled for this tensor.")
        self.grad = np.ones(shape=self.data.shape, dtype=self.data.dtype)
        nodes_to_process = [self]

        while nodes_to_process:
            current_node = nodes_to_process.pop()
            if current_node.inputs_node:
                if current_node.operation:
                    operation_class = getattr(autodiff.ops, current_node.operation.split('<')[1].strip('>'))
                    operation_instance = operation_class()

                    operation_instance.backward(current_node)
                    current_node.clip_grad()

                for input_node in current_node.inputs_node:
                    if input_node.requires_grad:
                        nodes_to_process.append(input_node)

    # =============== arithmetic operators and function implementation =================
    def __add__(self, other):
        return autodiff.add(o1=self, o2=other)

    def __sub__(self, other):
        return autodiff.subtract(o1=self, o2=other)

    def __mul__(self, other):
        return autodiff.mul(o1=self, o2=other)

    def matmul(self, other):
        return autodiff.matmul(o1=self, o2=other)

    def __truediv__(self, other):
        return autodiff.div(o1=self, o2=other)

    def reshape(self, shape):
        return autodiff.reshape(inp=self, shape=shape)

    def transpose(self):
        return autodiff.transpose(inp=self, axis=None)

    def flip(self, axis):
        return autodiff.flip(self=self, axis=axis)

    def summation(self, axis: None):
        return autodiff.summation(inp=self, axis=axis)

    @staticmethod
    def stack(dim, tensors):
        return autodiff.stack(dim, tensors)

    def __pow__(self, other):
        pass

    def sin(self):
        return autodiff.sin(op=self)

    def cos(self):
        return autodiff.cos(op=self)

    def mean(self):
        return autodiff.mean(inp=self)

    def log(self):
        return autodiff.log(inp=self)

    def exp(self):
        return autodiff.exp(inp=self)

    def tan(self):
        pass
