# ---------------- encoding utf-8 ------------------
from nn.module import Module
from autodiff.diff import Tensor
import numpy as np
import os
import sys

"""_summary_
    Module class is the base class for this whole nn.py
    Every nueral network class will inherit this Module class
"""
# this is jsut a simple Linear layer with in_feature and out_feature parameter
# similar to pytorch


class Linear(Module):
    def __init__(self, in_features, out_features, bias_option=True):
        super(Linear, self).__init__()
        self.bias_option = bias_option
        self.in_feature = in_features
        self.out_feature = out_features
        self.weight = Tensor(data=np.random.randn(in_features, out_features), requires_grad=True, dtype=np.float32)
        if self.bias_option:
            self.bias = Tensor(data=np.zeros(out_features), requires_grad=True, dtype=np.float32)
            self._parameters = {'weight': self.weight, 'bias': self.bias}
        else:
            self._parameters = {"weight": self.weight}

    def forward(self, x: Tensor):
        if self.bias_option:
            return x.matmul(other=self.weight) + self.bias
        else:
            return x.matmul(other=x)

    def __repr__(self) -> str:
        strg = f"nn.Linear{self.in_feature, self.out_feature}"
        return strg


class Embedding(Module):

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.num_embeddings = vocab_size
        self.dim = dim
        self.weight = Tensor(np.random.randn(self.num_embeddings, self.dim), requires_grad=True, dtype=np.float32)
        self._parameters['weight'] = self.weight

    def forward(self, idx):
        return Tensor(data=self.weight.data[idx], requires_grad=True, dtype=np.float32)

    def __repr__(self) -> str:
        strg = f"nn.Embedding{self.weight.data.shape}"
        return strg
