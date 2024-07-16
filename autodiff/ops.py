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

    def backward(self, out: Tensor):
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

    def backward(self, output_node: Tensor):
        o1, o2 = output_node.inputs_node
        grad = output_node.grad

        if o1.requires_grad:
            if o1.grad is None:
                o1.grad = self._sum_grad(grad, o1.data.shape)
            else:
                o1.grad += self._sum_grad(grad, o1.data.shape)

        if o2.requires_grad:
            if o2.grad is None:
                o2.grad = self._sum_grad(grad, o2.data.shape)
            else:
                o2.grad += self._sum_grad(grad, o2.data.shape)

    @staticmethod
    def _sum_grad(grad, shape):
        """Sum the gradient along the broadcasted dimensions."""
        while len(grad.shape) > len(shape):
            grad = np.sum(grad, axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad


class Negation:

    def forward(self, o1):
        o1.data = -1 * o1.data

    def backward(self, out):
        out.grad = -1 * out.grad


class SubtractionScalar:
    def forward(self, o1: Tensor, o2):
        out = o1.data - o2
        return Tensor(data=out, dtype=o1.data.dtype, requires_grad=o1.requires_grad, inputs_node=[o1], operation="Backward<SubtractionScalar>")

    def backward(self, out: Tensor):
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

    def backward(self, output_node: Tensor):
        o1, o2 = output_node.inputs_node
        grad = output_node.grad

        if o1.requires_grad:
            if o1.grad is None:
                o1.grad = -self._sum_grad(grad, o1.data.shape)
            else:
                o1.grad -= self._sum_grad(grad, o1.data.shape)

        if o2.requires_grad:
            if o2.grad is None:
                o2.grad = self._helper_grad(grad, o2.data.shape)
            else:
                o2.grad += self._helper_grad(grad, o2.data.shape)

    @staticmethod
    def _helper_grad(grad, shape):
        """Sum the gradient along the broadcasted dimensions."""
        while len(grad.shape) > len(shape):
            grad = np.sum(grad, axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad


class Reshape:
    def forward(self, inp: Tensor, shape: tuple):
        inp.data = np.reshape(a=inp.data, newshape=shape)

    def backward(self, inp: Tensor):
        inp.grad = np.reshape(a=inp.grad, newshape=inp.data.shape)

# ----------------- transpose and permute both are same for now


class Transpose:
    """_summary_
        this class define the forward and backward pass for the transpose function
        as transpose the data of tensor object similarly the backward of that
        tensor object will be transposed.
    """

    def forward(self, inp: Tensor, axis: None):
        inp.data = np.transpose(a=inp.data, axes=axis)

    def backward(self, inp: Tensor, axis: None):
        inp.grad = np.transpose(a=inp.grad, axes=axis)


class Permutation:
    """_summary_
        this class define the forward and backward pass for the pemute() function
        as permute the data of tensor object similarly the backward of that
        tensor object will be permuted.
    """

    def forward(self, inp: Tensor, axis=None):
        self.data = np.transpose(a=inp.data, axes=axis)

    def backward(self, out: Tensor, axis=None):
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

    def backward(self, output_node: Tensor):
        parmas = output_node.inputs_node[0]
        parmas.grad = np.ones_like(a=parmas.data)
        parmas.grad *= output_node.grad * (1/parmas.data)


class Exp:
    """_summary_

    Returns:
        _type_: Tensor
        it calculates the exponent of the tensor and returns the output tensor and also defines the backward pass
    """

    def forward(self, inp: Tensor):
        out = np.exp(inp.data)
        return Tensor(data=out, requires_grad=inp.requires_grad, dtype=inp.dtype, operation="Backward<Exp>", inputs_node=[inp])

    def backward(self, output_node: Tensor):
        param = output_node.inputs_node[0]
        param.grad = np.ones_like(a=param.data)
        param.grad *= output_node.grad * np.exp(param.data)


class Mean:
    """_summary_
        this class defines the forward and backward pass for the mean function on a  tensor object .
        grad also be taken as mean.
    """

    def forward(self, inp: Tensor):
        out = np.mean(a=inp.data)
        return Tensor(data=out, dtype=inp.dtype, requires_grad=inp.requires_grad, inputs_node=[inp], operation="Backward<Mean>")

    def backward(self, output_node: Tensor):
        param = output_node.inputs_node[0]
        param.grad = np.ones_like(a=param.data)
        param.grad *= output_node.grad/param.data.size


class MultiplicationScalar:
    def forward(self, op1, op2):  # op2 is scalar and op1 is tensor
        pass

    def backward(self, output_node):
        pass

# one edge case is remaining => broadcasting


class MultiplicationEWise:
    def forward(self, op1, op2):
        req_grad = False
        out = np.multiply(op1, op2)
        if op1.requires_grad:
            req_grad = True
        elif op2.requires_grad:
            req_grad = True
        return Tensor(data=out, requires_grad=req_grad, dtype=out.dtype, inputs_node=[op1, op2], operation="Backward<MultiplicationEWise>")

    def backward(self, output_node: Tensor):
        op1, op2 = output_node.inputs_node
        op1.grad = output_node.grad * np.full(shape=op2.data.shape, fill_value=op2.data)
        op2.grad = output_node.grad * np.full(shape=op1.data.shape, fill_value=op1.data)


class Multiplication:
    """_summary_
        this class handles forward and backward pass for the multiplication of nd tensor
    """

    def forward(self, op1, op2):
        req_grad = False
        out = np.matmul(op1.data, op2.data)
        if op1.requires_grad:
            req_grad = True
        elif op2.requires_grad:
            req_grad = True
        return Tensor(data=out, dtype=out.dtype, requires_grad=req_grad, operation="Backward<Multiplication>", inputs_node=[op1, op2])

    def backward(self, output_node: Tensor):
        op1, op2 = output_node.inputs_node
        grad = output_node.grad

        if op1.requires_grad:
            grad_op1 = np.matmul(grad, np.swapaxes(op2.data, -1, -2))
            op1.grad = self._accumulate_gradient(op1.grad, grad_op1, op1.data.shape)

        if op2.requires_grad:
            grad_op2 = np.matmul(np.swapaxes(op1.data, -1, -2), grad)
            op2.grad = self._accumulate_gradient(op2.grad, grad_op2, op2.data.shape)

    @staticmethod
    def _accumulate_gradient(existing_grad, new_grad, shape):
        """Sum the gradient along the broadcasted dimensions."""
        while len(new_grad.shape) > len(shape):
            new_grad = np.sum(new_grad, axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                new_grad = np.sum(new_grad, axis=i, keepdims=True)
        if existing_grad is None:
            return new_grad
        return existing_grad + new_grad


class DivisionEWise:

    def forward(self, op1, op2):
        req_grad = False
        out = op1 / op2
        if op1.requires_grad:
            req_grad = True
        elif op2.requires_grad:
            req_grad = True
        return Tensor(data=out, dtype=out.dtype, requires_grad=req_grad, inputs_node=[op1, op2], operation="Backward<DivisionEWise>")

    def backward(self, output_node: Tensor):
        op1, op2 = output_node.inputs_node
        op1.grad = np.zeros_like(a=op1.data)
        op2.grad = np.zeros_like(a=op2.data)

        op1.grad += output_node.grad / op2.data
        op2.grad += output_node.grad * (-op1.data / (op2.data ** 2))
