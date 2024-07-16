# ------------------ utf-8 encoding ------------------------
import numpy as np
import os
import sys
import math
import random
from autodiff.diff import Tensor


class Initializer:

    def randn(self, shape: tuple, dtype, requires_grad=False):
        data = np.random.rand(*shape)
        return Tensor(data=data, dtype=dtype, requires_grad=requires_grad)

    def arange(self, n1=0, n2=100, dtype=np.float32, requires_grad=False):
        data = np.arange(start=n1, step=n2, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def uniform(self, shape, requires_grad=False, low=0, high=1):
        data = np.random.uniform(size=shape, low=low, high=high)
        return Tensor(data=data, requires_grad=requires_grad, dtype=float)

    def normal(self, shape, requires_grad=False, loc=0, scale=0.5):
        data = np.random.normal(loc=loc, scale=scale, size=shape)
        return Tensor(data=data, requires_grad=requires_grad, dtype=float)
