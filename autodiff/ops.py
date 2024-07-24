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
                o1.grad = -self._helper_grad(grad, o1.data.shape)
            else:
                o1.grad -= self._helper_grad(grad, o1.data.shape)

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
    def forward(self, inp: Tensor, shape: tuple) -> Tensor:
        reshaped_data = np.reshape(a=inp.data, newshape=shape)
        return Tensor(
            data=reshaped_data,
            requires_grad=inp.requires_grad,
            dtype=inp.dtype,
            inputs_node=[inp],
            operation="Backward<Reshape>"
        )

    def backward(self, output_node: Tensor):
        input_node = output_node.inputs_node[0]
        input_node.grad = np.reshape(a=output_node.grad, newshape=input_node.data.shape)


# ----------------- transpose and permute both are same for now


class Transpose:
    """_summary_
        This class defines the forward and backward pass for the transpose function.
        The forward pass transposes the data of a Tensor object.
        The backward pass transposes the gradient of that Tensor object using the same axes.
    """

    def forward(self, inp: Tensor, axis=None):
        self.axis = axis
        data = np.transpose(a=inp.data, axes=axis)
        out = Tensor(data=data, requires_grad=inp.requires_grad, dtype=inp.dtype,
                     inputs_node=[inp], operation="Backward<Transpose>")
        out.axis = axis
        return out

    def backward(self, output_node: Tensor):
        inp = output_node.inputs_node[0]
        inp.grad = np.transpose(a=output_node.grad, axes=output_node.axis if output_node.axis else None)


class Stack:
    dim = None

    def __init__(self) -> None:
        pass

    def forward(self, dim, tensors):
        self.dim = dim
        requires_grad = any(tensor.requires_grad for tensor in tensors)
        data = np.stack([tensor.data for tensor in tensors], axis=self.dim)
        return Tensor(data=data, requires_grad=requires_grad, dtype=data.dtype, inputs_node=tensors, operation="Backward<Stack>", axis=self.dim)

    def backward(self, output_node: Tensor):
        tensors = output_node.inputs_node
        grads = np.split(output_node.grad, indices_or_sections=len(tensors), axis=output_node.axis)
        for tensor, grad_part in zip(tensors, grads):
            tensor.grad = grad_part.squeeze(self.dim)


class Split:
    def __init__(self):
        pass

    def forward(self, tensor, axis, indices_or_sections):
        self.axis = axis
        self.tensor_shape = tensor.data.shape
        self.indices_or_sections = indices_or_sections
        requires_grad = tensor.requires_grad
        split_data = np.split(tensor.data, indices_or_sections, axis=axis)

        split_tensors = [
            Tensor(data, np.float32, True, "Backward<Split>", [tensor], self.axis, [self.tensor_shape])
            for data in split_data
        ]

        return split_tensors

    def backward(self, output_nodes):
        for it in output_nodes:
            it.grad = np.random.randn(*it.data.shape)
        original_tensor = output_nodes[0].inputs_node[0]
        grad = np.zeros_like(original_tensor.data)
        grads = [output_node.grad for output_node in output_nodes]
        for i, grad_part in enumerate(grads):
            expanded_grad = np.expand_dims(grad_part, axis=output_nodes[0].axis)
            grad_slices = [slice(None)] * len(output_nodes[0].params[0])
            grad_slices[output_nodes[0].axis] = slice(
                i * grad_part.shape[output_nodes[0].axis], (i + 1) * grad_part.shape[output_nodes[0].axis])
            grad[tuple(grad_slices)] += expanded_grad.squeeze()

        if original_tensor.grad is None:
            original_tensor.grad = grad
        else:
            original_tensor.grad += grad


class Slice:
    def forward(self, inp_tensor: Tensor, idx):
        sliced_data = inp_tensor.data[idx]
        if inp_tensor.requires_grad:
            return Tensor(data=sliced_data, requires_grad=True, dtype=inp_tensor.dtype, inputs_node=[inp_tensor], operation="Backward<Slice>", slice_indices=idx)
        else:
            return Tensor(data=sliced_data, requires_grad=False, dtype=inp_tensor.dtype)

    def backward(self, output_node: Tensor):
        inp_tensor = output_node.inputs_node[0]
        if inp_tensor.grad is None:
            inp_tensor.grad = np.zeros_like(inp_tensor.data)
        np.add.at(inp_tensor.grad, output_node.slice_indices, output_node.grad)


class Permutation:
    """_summary_
        this class define the forward and backward pass for the pemute() function
        as permute the data of tensor object similarly the backward of that
        tensor object will be permuted.
    """
    axis = None

    def forward(self, inp: Tensor, axis=None):
        self.axis = axis
        self.data = np.transpose(a=inp.data, axes=self.axis)

    def backward(self, out: Tensor):
        out.grad = np.transpose(a=out.grad, axes=self.axis)


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


class Sin:
    def forward(self, inp: Tensor):
        out = np.sin(inp.data)
        return Tensor(data=out, requires_grad=inp.requires_grad, dtype=out.dtype, inputs_node=[inp], operation="Backward<Sin>")

    def backward(self, output_node: Tensor):
        inputs = output_node.inputs_node[0]
        inputs.grad = np.cos(inputs.data) * output_node


class Cos:
    def forward(self, inp: Tensor):
        out = np.cos(inp.data)
        return Tensor(data=out, requires_grad=inp.requires_grad, dtype=out.dtype, inputs_node=[inp], operation="Backward<Cos>")

    def backward(self, output_node: Tensor):
        inputs = output_node.inputs_node[0]
        inputs.grad = -np.sin(inputs.data) * output_node


class Max:
    def forward(self, inp_tensor: Tensor, axis=None, keep_dims=False):
        self.axis = axis
        self.keep_dims = keep_dims
        out = np.max(a=inp_tensor.data, axis=self.axis, keepdims=self.keep_dims)
        return Tensor(out, out.dtype, inp_tensor.requires_grad, "Backward<Max>",  [inp_tensor], self.axis, [self.keep_dims])

    def backward(self, output_node: Tensor):
        input_tensor = output_node.inputs_node[0]
        keep_dims = output_node.params[0]
        axis = output_node.axis

        grad = output_node.grad
        expanded_grad = np.expand_dims(grad, axis=axis) if not keep_dims else grad

        max_mask = (input_tensor.data == np.expand_dims(output_node.data,
                    axis=axis) if not keep_dims else output_node.data)
        input_tensor.grad = max_mask * expanded_grad


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


class Concatenate:
    """_summary_
        This class defines the forward and backward pass for concatenating
        multiple tensor objects along a specified axis.
    """

    def __init__(self):
        pass

    def forward(self, inputs, axis=0):
        self.axis = axis
        out = np.concatenate([inp.data for inp in inputs], axis=self.axis)
        return Tensor(data=out, dtype=inputs[0].dtype, requires_grad=any(inp.requires_grad for inp in inputs),
                      inputs_node=inputs, operation="Backward<Concatenate>", axis=self.axis)

    def backward(self, output_node: Tensor):
        grad = output_node.grad
        input_grads = []

        for inp in output_node.inputs_node:
            input_grad = np.sum(grad, axis=output_node.axis, keepdims=True) * np.ones_like(inp.data)
            input_grads.append(input_grad)

        for i, inp in enumerate(output_node.inputs_node):
            if inp.requires_grad:
                inp.grad = input_grads[i]


class MultiplicationScalar:
    def forward(self, op1, op2):  # op2 is scalar and op1 is tensor
        pass

    def backward(self, output_node):
        pass

# one edge case is remaining => broadcasting


class MultiplicationEWise:
    def forward(self, op1, op2):
        out_data = np.multiply(op1.data, op2.data)
        req_grad = op1.requires_grad or op2.requires_grad
        out = Tensor(data=out_data, requires_grad=req_grad, dtype=out_data.dtype,
                     inputs_node=[op1, op2], operation="Backward<MultiplicationEWise>")
        return out

    def backward(self, output_node: Tensor):
        op1, op2 = output_node.inputs_node
        grad = output_node.grad

        # Handling the gradient for op1
        if op1.requires_grad:
            grad_op1 = grad * op2.data

            # Sum gradients over appropriate axes
            while len(grad_op1.shape) > len(op1.data.shape):
                grad_op1 = np.sum(grad_op1, axis=0)
            for axis, size in enumerate(op1.data.shape):
                if size == 1:
                    grad_op1 = np.sum(grad_op1, axis=axis, keepdims=True)

            if op1.grad is None:
                op1.grad = grad_op1
            else:
                op1.grad += grad_op1

        # Handling the gradient for op2
        if op2.requires_grad:
            grad_op2 = grad * op1.data

            # Sum gradients over appropriate axes
            while len(grad_op2.shape) > len(op2.data.shape):
                grad_op2 = np.sum(grad_op2, axis=0)
            for axis, size in enumerate(op2.data.shape):
                if size == 1:
                    grad_op2 = np.sum(grad_op2, axis=axis, keepdims=True)

            if op2.grad is None:
                op2.grad = grad_op2
            else:
                op2.grad += grad_op2


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


class Summation:
    axis = None

    def forward(self, inp: Tensor, axis=None):
        self.axis = axis
        out = np.sum(a=inp.data, axis=self.axis, dtype=inp.data.dtype, keepdims=True)
        return Tensor(data=out, dtype=inp.dtype, requires_grad=inp.requires_grad, inputs_node=[inp], operation="Backward<Summation>", axis=self.axis)

    def backward(self, output_node: Tensor):
        o1 = output_node.inputs_node[0]
        if o1.requires_grad:
            grad_output = output_node.grad
            # Expand the output gradient to match the input tensor's shape
            if self.axis is not None:
                grad_output = np.expand_dims(grad_output, axis=output_node.axis)
                repeat_dims = o1.data.shape
                grad_output = np.broadcast_to(grad_output, repeat_dims)

            if o1.grad is None:
                o1.grad = grad_output
            else:
                o1.grad += grad_output


class Sqrt:

    def forward(self, inp_tensor: Tensor):
        out = np.sqrt(inp_tensor.data)
        return Tensor(data=out, dtype=out.dtype, requires_grad=inp_tensor.requires_grad, operation="Backward<Sqrt>", inputs_node=[inp_tensor])

    def backward(self, output_node: Tensor):
        inputs_node = output_node.inputs_node[0]
        inputs_node.grad = output_node.grad/2 * (1/np.sqrt(inputs_node.data))


class Flip:
    def forward(self, op: Tensor, axis):
        data = np.flip(op.data, axis=axis)
        return Tensor(data=data, requires_grad=op.requires_grad, dtype=op.dtype, inputs_node=[op], operation="Backward<Flip>")

    def backward(self, output_node: Tensor):
        param = output_node.inputs_node[0]
        param.grad = np.full(shape=param.data.shape, fill_value=1, dtype=np.float32)
        param.grad = np.flip(m=output_node.grad, axis=output_node.axis)


class Dilation:
    def forward(self, inp, dilation_factor):
        self.diletion_factor = dilation_factor
        self.input_shape = inp.data.shape
        dilated_shape = [dim * self.dilation_factor for dim in self.input_shape]
        dilated_data = np.zeros(dilated_shape, dtype=inp.data.dtype)

        slices = [slice(None, None, self.dilation_factor) for _ in range(len(self.input_shape))]
        dilated_data[tuple(slices)] = inp.data

        return Tensor(data=dilated_data, requires_grad=inp.requires_grad, operation="Backward<Dilation>", inputs_node=[inp], axis=self.diletion_factor)

    def backward(self, output_node):
        inp = output_node.inputs_node[0]
        if inp.grad is None:
            inp.grad = np.zeros_like(inp.data)

        slices = [slice(None, None, output_node.dilation_factor) for _ in range(len(self.input_shape))]
        inp.grad += output_node.grad[tuple(slices)]


class Undilation:
    def __init__(self, dilation_factor):
        pass

    def forward(self, inp, dilation_factor):
        self.dilation_factor = dilation_factor
        self.input_shape = inp.data.shape
        undilated_shape = [dim // self.dilation_factor for dim in self.input_shape]
        undilated_data = inp.data[tuple(slice(None, None, self.dilation_factor) for _ in range(len(self.input_shape)))]

        return Tensor(data=undilated_data, requires_grad=inp.requires_grad, operation="Backward<Undilation>", inputs_node=[inp], axis=self.dilation_factor)

    def backward(self, output_node):
        inp = output_node.inputs_node[0]
        if inp.grad is None:
            inp.grad = np.zeros_like(inp.data)

        slices = [slice(None, None, output_node.dilation_factor) for _ in range(len(self.input_shape))]
        np.add.at(inp.grad, tuple(slices), output_node.grad)


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


class Softmax:
    """_summary_
        This class defines the forward and backward pass for the softmax function.
        The forward pass computes the softmax of the input tensor data.
        The backward pass computes the gradient of the softmax output.
    """

    def __init__(self):
        self.axis = -1

    def forward(self, inp_tensor: Tensor, dim: int):
        exps = np.exp(inp_tensor.data - np.max(inp_tensor.data, axis=dim, keepdims=True))
        softmax_output = exps / np.sum(exps, axis=dim, keepdims=True)
        out = Tensor(data=softmax_output, requires_grad=inp_tensor.requires_grad,
                     dtype=inp_tensor.dtype, inputs_node=[inp_tensor], operation="Backward<Softmax>", axis=dim)
        return out

    def backward(self, output_node: Tensor):
        inp = output_node.inputs_node[0]
        grad_output = output_node.grad
        softmax_output = output_node.data
        s = softmax_output.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        inp.grad = np.dot(jacobian, grad_output)


class Flatten:

    def forward(self, inp_tensor: Tensor):
        data = inp_tensor.data.flatten()
        return Tensor(data=data, requires_grad=inp_tensor.requires_grad, dtype=inp_tensor.data.dtype, operation="Backward<Flatten>", inputs_node=[inp_tensor])

    def backward(self, output_node: Tensor):
        input_node: Tensor = output_node.inputs_node[0]
        input_node.grad = output_node.grad.reshape(input_node.data.shape)


class Std:
    """
    This class defines the forward and backward pass for the standard deviation function on a Tensor object.
    Gradients are also computed with respect to the mean.
    """

    def forward(self, inp_tensor: Tensor):
        mean = np.mean(inp_tensor.data)
        variance = np.mean((inp_tensor.data - mean) ** 2)
        std_dev = np.sqrt(variance)
        return Tensor(
            data=std_dev,
            dtype=inp_tensor.data.dtype,
            requires_grad=inp_tensor.requires_grad,
            inputs_node=[inp_tensor],
            operation="Backward<StandardDeviation>"
        )

    def backward(self, output_node: Tensor):
        inp = output_node.inputs_node[0]
        mean = np.mean(inp.data)
        variance = np.mean((inp.data - mean) ** 2)
        std_dev = np.sqrt(variance)

        # Gradient of the standard deviation with respect to the input
        grad = (inp.data - mean) / (std_dev * inp.data.size)
        inp.grad = grad * output_node.grad
