# ---------------- encoding utf-8 ------------------
from nn.module import Module
from autodiff.diff import Tensor
from initializes.random_init import Initializer
from nn.functonal import sigmoid, relu, tanh
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


class RNNCell(Module):
    """
    weight_ih (torch.Tensor) = the learnable input-hidden weights, of shape (hidden_size, input_size)
    weight_hh (torch.Tensor) = the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
    bias_ih = the learnable input-hidden bias, of shape (hidden_size)
    bias_hh = the learnable hidden-hidden bias, of shape (hidden_size)

    """

    def __init__(self, input_size, hidden_size, bias_option, non_linear_act) -> None:
        rinit = Initializer()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option
        self.non_linearity = non_linear_act
        self.w_ih = rinit.lecun_uniform(shape=(self.hidden_size, self.input_size),
                                        n_in=self.hidden_size, requires_grad=True)
        self.w_hh = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                        n_in=self.hidden_size, requires_grad=True)
        if bias_option:
            self.bias_ih = rinit.lecun_uniform(
                shape=(self.hidden_size, 1), n_in=self.hidden_size, requires_grad=True)
            self.bias_hh = rinit.lecun_uniform(
                shape=(self.hidden_size, 1), n_in=self.hidden_size, requires_grad=True)
            self._parameters = {'weight_ih': self.w_ih, 'bias_ih': self.bias_ih,
                                "weight_hh": self.w_hh, "bias_hh": self.bias_hh}
        else:
            self._parameters = {'weight_ih': self.w_ih, "weight_hh": self.w_hh}

    def forward(self, x: Tensor, hx: Tensor):
        print("x shape is ", x.shape())
        print("hx shape is ", hx.shape())
        print("w_ih shape is ", self.w_ih.shape())
        print("w_hh shape is ", self.w_hh.shape())
        if self.bias_option:
            x.transpose()
            hx.transpose()
            h_new = self.w_ih.matmul(x) + self.bias_ih + self.w_hh.matmul(hx) + self.bias_hh
            if self.non_linearity == "relu":
                h_out = relu(inp_tensor=h_new)
                return h_out
            else:
                h_out = sigmoid(inp_tensor=h_new)
                return h_out
        else:
            h_new = self.w_ih.matmul(x) + self.w_hh.matmul(hx)
            if self.non_linearity == "relu":
                h_out = relu(inp_tensor=h_new)
                return h_out
            else:
                h_out = sigmoid(inp_tensor=h_new)
                return h_out

    def __repr__(self):
        strg = f"nn.RNNCell{self.input_size,self.hidden_size}"
        return strg


class GRUCell(Module):
    """
            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
            h' = (1 - z) \odot n + z \odot h
            \end{array}

    """

    def __init__(self, input_size, hidden_size, bias_option, ) -> None:
        rinit = Initializer()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option

        # all three gates that used in grucell
        self.reset_gate = Linear(in_features=self.input_size+self.hidden_size,
                                 out_features=self.hidden_size, bias_option=True)
        self.forget_gate = Linear(in_features=self.input_size+self.hidden_size,
                                  out_features=self.hidden_size, bias_option=True)
        self.update_gate = Linear(in_features=self.input_size+self.hidden_size,
                                  out_features=self.hidden_size, bias_option=True)

        self.add_module["reset_gate"] = self.reset_gate
        self.add_module["forget_gate"] = self.forget_gate
        self.add_module["update_gate"] = self.update_gate

    def forward(self, x: Tensor, hx: Tensor):
        out = Tensor.cat(inputs=[x, hx], axis=1)
        r_out = sigmoid(inp_tensor=self.reset_gate(out))
        u_out = sigmoid(inp_tensor=self.update_gate(out))
        f_out = sigmoid(inp_tensor=self.forget_gate(Tensor.cat([x, r_out * hx], axis=1)))
        h_new = (Tensor(data=1, requires_grad=True, dtype=np.float32) - u_out) * (f_out + (u_out * hx))
        return h_new


class LSTMCell:

    def __init__(self) -> None:
        pass

    def forward(self, x):
        pass
