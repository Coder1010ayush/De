# -------------- utf-8 encoding --------------
import autodiff
import autodiff.diff
from autodiff.ops import AddEWise, AddScalar, SubEWise, SubtractionScalar, Log, Permutation, Transpose, Negation, Reshape, Mean


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
