# -------------- utf-8 encoding --------------
import autodiff
import autodiff.diff
from autodiff.ops import AddEWise, AddScalar, SubEWise, SubtractionScalar, Log, Permutation, Transpose, Negation, Reshape, Mean, Exp, Multiplication, MultiplicationEWise, MultiplicationScalar, DivisionEWise, Summation, Stack, Sin, Cos, Flip, Max, Dilation, Undilation, Concatenate, Split


def add(o1, o2):
    if isinstance(o2, autodiff.diff.Tensor):
        out = AddEWise().forward(o1=o1, o2=o2)
        return out
    else:
        out = AddScalar().forward(o1=o1, o2=o2)
        return out


def subtract(o1, o2):
    if isinstance(o2, autodiff.diff.Tensor):
        out = SubEWise().forward(o1=o1, o2=o2)
        return out
    else:
        out = SubtractionScalar().forward(o1=o1, o2=o2)
        return out


def mul(o1, o2):
    if isinstance(o2, float):
        out = MultiplicationScalar().forward(op1=o1, op2=o2)
        return out
    else:
        out = MultiplicationEWise().forward(op1=o1, op2=o2)
        return out


@staticmethod
def cat(inputs, axis):
    return Concatenate().forward(inputs=inputs, axis=axis)


def flip(self, axis):
    return Flip().forward(op=self, axis=axis)


def max(self, axis, keep_dims):
    return Max().forward(inp_tensor=self, axis=axis, keep_dims=keep_dims)


def dilate(self, dilate_factor):
    return Dilation().forward(inp=self, dilation_factor=dilate_factor)


def undilate(self, dilate_factor):
    return Undilation().forward(inp=self, dilation_factor=dilate_factor)


def stack(dim, tensors):
    obj = Stack()
    return obj.forward(dim=dim, tensors=tensors)


def split(inputs, axis, indices_or_sections):
    return Split().forward(tensor=inputs, axis=axis, indices_or_sections=indices_or_sections)


def matmul(o1, o2):
    out = Multiplication().forward(op1=o1, op2=o2)
    return out


def sin(op):
    return Sin().forward(inp=op)


def cos(op):
    return Cos().forward(inp=op)


def div(o1, o2):
    out = DivisionEWise().forward(op1=o1, op2=o2)
    return out


def reshape(inp, shape):
    return Reshape().forward(inp=inp, shape=shape)


def transpose(inp, axis: None):
    return Transpose().forward(inp=inp, axis=axis)


def permute(inp, axis: None):
    return Permutation().forward(inp=inp, axis=axis)


def log(inp):
    return Log().forward(o1=inp)


def mean(inp):
    return Mean().forward(inp=inp)


def exp(inp):
    return Exp().forward(inp=inp)


def summation(inp, axis=None):
    return Summation().forward(inp=inp, axis=axis)
