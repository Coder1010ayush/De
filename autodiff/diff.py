# ------- utf-8 encoding -------------
import autodiff
import os
import sys
import numpy as np
from utility import data_to_numpy
TENSOR_COUNTER = 0
mode = "forward"


class Tensor:

    def __init__(self, data, dtype, requires_grad=False, operation=None, inputs_node=[]) -> None:
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

    def shape(self):
        return self.data.shape

    # how tensor will be looking when printing in the console

    def __repr__(self) -> str:
        srepr = f"Tensor({self.data},{self.dtype})"
        return srepr

    # backpropogation function
    def backpropogate(self):
        if not self.requires_grad:
            raise ValueError("Gradient tracking is not enabled for this tensor.")

        self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        nodes_to_process = [self]

        while nodes_to_process:
            current_node = nodes_to_process.pop()

            if current_node.inputs_node:
                if current_node.operation:
                    operation_class = getattr(autodiff.ops, current_node.operation.split('<')[1].strip('>'))
                    operation_instance = operation_class()

                    operation_instance.gradient(current_node)

                for input_node in current_node.inputs_node:
                    if input_node.requires_grad:
                        nodes_to_process.append(input_node)

    # =============== arithmetic operators and function implementation =================
    def __add__(self, other):
        return autodiff.add(o1=self, o2=other)

    def __sub__(self, other):
        return autodiff.subtract(o1=self, o2=other)

    def __mul__(self, other):
        pass

    def __truediv(self, other):
        pass

    def __pow__(self, other):
        pass

    def sum(self):
        pass

    def mean(self):
        return autodiff.mean(inp=self)

    def log(self):
        return autodiff.log(inp=self)

    def exp(self):
        pass

    def sin(self):
        pass

    def cos(self):
        pass

    def tan(self):
        pass

    def sinh(self):
        pass

    def consh(self):
        pass

    def tanh(self):
        pass
