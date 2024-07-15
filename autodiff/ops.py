"""_summary_

    Defining classes for the all kinds of operators and functions that 
    can be used for the tensor object.
    
"""
import numpy as np
import os
import sys
from autodiff.diff import Tensor


class AddScalar:
    def forward(self, o1: Tensor, o2):
        out = o1.data + o2
        return Tensor(data=out, dtype=o1.data.dtype, requires_grad=o1.requires_grad, inputs_node=[o1], operation="Backward<AddScalar>")

    def gradient(self, out: Tensor):
        o1 = out.inputs_node
        o1.grad = out.grad


class AddEWise:
    def forward(self, o1, o2):
        req_grad = False
        out = np.add(o1.data, o2.data)
        if o1.requires_grad:
            req_grad = True
        elif o2.requires_grad:
            req_grad = True
        return Tensor(data=out, dtype=out.dtype, requires_grad=req_grad,  operation="Backward<AddEWise>", inputs_node=[o1, o2])

    def gradient(self, output_node: Tensor):
        o1, o2 = output_node.inputs_node
        o1.grad = np.full_like(o1.data, output_node.grad)
        o2.grad = np.full_like(o2.data, output_node.grad)


class Negation:

    def forward(self, o1):
        o1.data = -1 * o1.data

    def gradient(self, out):
        out.grad = -1 * out.grad


class SubtractionScalar:
    def forward(self, o1: Tensor, o2):
        out = o1.data - o2
        return Tensor(data=out, dtype=o1.data.dtype, requires_grad=o1.requires_grad, inputs_node=[o1], operation="Backward<SubtractionScalar>")

    def gradient(self, out: Tensor):
        o1 = out.inputs_node
        o1.grad = out.grad


class SubEWise:
    def forward(self, o1, o2):
        req_grad = False
        out = np.subtract(o1.data, o2.data)
        if o1.requires_grad:
            req_grad = True
        elif o2.requires_grad:
            req_grad = True
        return Tensor(data=out, dtype=out.dtype, requires_grad=req_grad,  operation="Backward<SubEWise>", inputs_node=[o1, o2])

    def gradient(self, output_node: Tensor):
        o1, o2 = output_node.inputs_node
        o1.grad = np.full_like(o1.data, -output_node.grad)
        o2.grad = np.full_like(o2.data, output_node.grad)


class Reshape:
    def forward(self, inp: Tensor, shape: tuple):
        inp.data = np.reshape(a=inp.data, newshape=shape)

    def gradient(self, inp: Tensor):
        inp.grad = np.reshape(a=inp.grad, newshape=inp.data.shape)

# ----------------- transpose and permute both are same for now


class Transpose:
    """_summary_
        this class define the forward and backward pass for the transpose function
        as transpose the data of tensor object similarly the gradient of that 
        tensor object will be transposed.
    """

    def forward(self, inp: Tensor, axis: None):
        inp.data = np.transpose(a=inp.data, axes=axis)

    def gradient(self, inp: Tensor, axis: None):
        inp.grad = np.transpose(a=inp.grad, axes=axis)


class Permutation:
    """_summary_ 
        this class define the forward and backward pass for the pemute() function
        as permute the data of tensor object similarly the gradient of that 
        tensor object will be permuted.
    """

    def forward(self, inp: Tensor, axis=None):
        self.data = np.transpose(a=inp.data, axes=axis)

    def gradient(self, out: Tensor, axis=None):
        out.grad = np.transpose(a=out.grad, axes=axis)


class Log:
    """_summary_
        f(x) = log(x) 
        f'(x) = 1/x 
        similarly we expand it for matrix's each element.
    """

    def forward(self, o1: Tensor):
        out = np.log(o1.data)
        return Tensor(data=out, dtype=o1.dtype, requires_grad=o1.requires_grad, inputs_node=[o1], operation="Backward<Log>")

    def gradient(self, output_node: Tensor):
        parmas = output_node.inputs_node[0]
        parmas.grad = np.ones_like(a=parmas.data)
        parmas.grad *= output_node.grad * (1/parmas.data)


class Mean:
    """_summary_
        this class defines the forward and backward pass for the mean function on a  tensor object .
        grad also be taken as mean.
    """

    def forward(self, inp: Tensor):
        out = np.mean(a=inp.data)
        return Tensor(data=out, dtype=inp.dtype, requires_grad=inp.requires_grad, inputs_node=[inp], operation="Backward<Mean>")

    def gradient(self, output_node: Tensor):
        param = output_node.inputs_node[0]
        param.grad = np.ones_like(a=param.data)
        param.grad *= output_node.grad/param.data.size
