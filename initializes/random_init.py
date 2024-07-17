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

    def rand(self, shape, dtype, mean, std, requires_grad=False):
        data = np.random.rand(*shape)*std + mean
        return Tensor(data=data, requires_grad=requires_grad, dtype=data.dtype)

    def identity(self, n, dtype, requires_grad=False):
        data = np.identity(n=n, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def ones(self, shape, dtype, requires_grad=False):
        data = np.ones(shape=shape, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def zeros(self, shape, dtype, requires_grad=False):
        data = np.zeros(shape=shape, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def constants(self, shape, val, dtype, requires_grad=False):
        data = np.full(shape=shape, fill_value=val, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def arange(self, n1=0, n2=100, dtype=np.float32, requires_grad=False):
        data = np.arange(start=n1, step=n2, dtype=dtype)
        return Tensor(data=data, requires_grad=requires_grad, dtype=dtype)

    def uniform(self, shape, requires_grad=False, low=0, high=1):
        data = np.random.uniform(size=shape, low=low, high=high)
        return Tensor(data=data, requires_grad=requires_grad, dtype=float)

    def normal(self, shape, requires_grad=False, loc=0, scale=0.5):
        data = np.random.normal(loc=loc, scale=scale, size=shape)
        return Tensor(data=data, requires_grad=requires_grad, dtype=float)

    def xaviar_uniform(self, shape, n_in, n_out, requires_grad=False):
        data = np.random.uniform(low=-np.sqrt(6/(n_in+n_out)), high=np.sqrt(6/(n_in+n_out)), size=shape)
        return Tensor(data=data, dtype=data.dtype, requires_grad=requires_grad)

    def xavier_normal(self, shape, n_in, n_out, requires_grad=False):
        data = np.random.normal(size=shape, loc=0, scale=2/(n_in+n_out))
        return Tensor(data=data, requires_grad=requires_grad, dtype=data.dtype)

    def lecun_uniform(self, shape, n_in, requires_grad=False):
        data = np.random.uniform(size=shape, low=-np.sqrt(1/n_in), high=np.sqrt(1/n_in))
        return Tensor(data=data, requires_grad=requires_grad, dtype=data.dtype)

    def lecun_normal(self, shape, n_in, requires_grad=False):
        data = np.random.normal(size=shape, loc=0, scale=1/n_in)
        return Tensor(data=data, requires_grad=requires_grad, dtype=data.dtype)
