# ---------------- encoding utf-8 ------------------
from nn.module import Module
from autodiff.diff import Tensor
from autodiff import cat
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
            self.add_parameter(name=f"bias", value=self.bias)
            self.add_parameter(name=f"weights", value=self.weight)
        else:
            self.add_parameter(name=f"weights", value=self.weight)

    def forward(self, x: Tensor):
        if self.bias_option:
            return x.matmul(other=self._parameters["weights"].value) + self._parameters["bias"].value
        else:
            return x.matmul(other=self._parameters["weights"].value)

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
        self.add_parameter(name=f"weights", value=self.weight)

    def forward(self, idx):
        return Tensor(data=self._parameters["weights"].value[idx], requires_grad=True, dtype=np.float32)

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
        self.w_ih = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                        n_in=self.hidden_size, requires_grad=True)
        self.w_hh = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                        n_in=self.hidden_size, requires_grad=True)
        if bias_option:
            self.bias_ih = rinit.lecun_uniform(
                shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)
            self.bias_hh = rinit.lecun_uniform(
                shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)
            self.add_parameter(name="w_ih", value=self.w_ih)
            self.add_parameter(name="w_hh", value=self.w_hh)
            self.add_parameter(name="b_ih", value=self.bias_ih)
            self.add_parameter(name="b_hh", value=self.bias_hh)

        else:
            self.add_parameter(name="w_ih", value=self.w_ih)
            self.add_parameter(name="w_hh", value=self.w_hh)

    def forward(self, x: Tensor, hx: Tensor):
        if self.bias_option:
            h_new = x.matmul(self._parameters["w_ih"].value) + \
                self._parameters["b_ih"].value + \
                hx.matmul(self._parameters["w_hh"].value) + self._parameters["b_hh"].value
            if self.non_linearity == "relu":
                h_out = relu(inp_tensor=h_new)
                return h_out
            else:
                h_out = sigmoid(inp_tensor=h_new)
                return h_out
        else:
            h_new = self._parameters["w_ih"].value.matmul(x) + self._parameters["w_hh"].value.matmul(hx)
            if self.non_linearity == "relu":
                h_out = relu(inp_tensor=h_new)
                return h_out
            else:
                h_out = sigmoid(inp_tensor=h_new)
                return h_out

    def __repr__(self):
        strg = f"nn.RNNCell{self.input_size,self.hidden_size}"
        return strg


class RNN:
    def __init__(self) -> None:
        pass


class GRUCell(Module):
    """
            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
            h' = (1 - z) \odot n + z \odot h
            \end{array}

    """

    def __init__(self, input_size, hidden_size, bias_option) -> None:
        super().__init__()
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

        self.add_module("reset_gate",  self.reset_gate)
        self.add_module("forget_gate", self.forget_gate)
        self.add_module("update_gate", self.update_gate)

    def forward(self, x: Tensor, hx: Tensor):
        ls = []
        ls.append(x)
        ls.append(hx)
        out = cat(inputs=ls, axis=1)
        r_out = sigmoid(inp_tensor=self.reset_gate(out))
        u_out = sigmoid(inp_tensor=self.update_gate(out))
        print("rout is ", r_out)
        print(" --------------------------------------- ")
        print("rout shape is ", r_out.shape(), type(r_out))
        print("hx shape is ", hx.shape(), type(hx))
        intermediate_out = r_out * hx
        f_out = sigmoid(inp_tensor=self.forget_gate(cat([x, intermediate_out], axis=1)))
        h_new = (Tensor(data=1, requires_grad=True, dtype=np.float32) - u_out) * (f_out + (u_out * hx))
        return h_new


class LSTMCell(Module):

    def __init__(self, input_size: int, hidden_size: int, bias_option: bool = True) -> None:
        rinit = Initializer()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option
        """
            there are four gates in the lstm cell each gate have its 
            own weight and bias 
        """
        # forget gate
        if self.bias_option:
            self.w_if = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hf = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.b_f = rinit.lecun_normal(shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)

            # input gate
            self.w_ii = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hi = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.b_i = rinit.lecun_normal(shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)

            # g gate
            self.w_ig = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hg = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.b_g = rinit.lecun_normal(shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)

            # o gate
            self.w_io = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_ho = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.b_o = rinit.lecun_normal(shape=(1, self.hidden_size), n_in=self.hidden_size, requires_grad=True)
            self.add_parameter(name="w_if", value=self.w_if)
            self.add_parameter(name="w_hf", value=self.w_hf)
            self.add_parameter(name="w_ii", value=self.w_ii)
            self.add_parameter(name="w_hi", value=self.w_hi)
            self.add_parameter(name="w_ig", value=self.w_ig)
            self.add_parameter(name="w_hg", value=self.w_hg)
            self.add_parameter(name="w_io", value=self.w_io)
            self.add_parameter(name="w_ho", value=self.w_ho)
            self.add_parameter("b_o", self.b_o)
            self.add_parameter("b_g", self.b_g)
            self.add_parameter("b_i", self.b_i)
            self.add_parameter("b_f", self.b_f)
        else:
            self.w_if = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hf = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            # input gate
            self.w_ii = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hi = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            # g gate
            self.w_ig = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_hg = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            # o gate
            self.w_io = rinit.lecun_uniform(shape=(self.input_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.w_ho = rinit.lecun_uniform(shape=(self.hidden_size, self.hidden_size),
                                            n_in=self.hidden_size, requires_grad=True)
            self.add_parameter(name="w_if", value=self.w_if)
            self.add_parameter(name="w_hf", value=self.w_hf)
            self.add_parameter(name="w_ii", value=self.w_ii)
            self.add_parameter(name="w_hi", value=self.w_hi)
            self.add_parameter(name="w_ig", value=self.w_ig)
            self.add_parameter(name="w_hg", value=self.w_hg)
            self.add_parameter(name="w_io", value=self.w_io)
            self.add_parameter(name="w_ho", value=self.w_ho)

    def forward(self, x: Tensor, hx: list):
        h_prev = hx[0]
        c_prev = hx[1]
        if self.bias_option:
            i_out = x.matmul(self._parameters["w_ii"].value) + \
                self._parameters["b_i"].value + h_prev.matmul(self._parameters["w_hi"].value)
            f_out = x.matmul(self._parameters["w_if"].value) + \
                self._parameters["b_f"].value + h_prev.matmul(self._parameters["w_hf"].value)
            g_out = x.matmul(self._parameters["w_ig"].value) + \
                self._parameters["b_g"].value + h_prev.matmul(self._parameters["w_hg"].value)
            o_out = x.matmul(self._parameters["w_io"].value) + \
                self._parameters["b_o"].value + h_prev.matmul(self._parameters["w_ho"].value)
            out = (f_out * c_prev) + (i_out * g_out)
            h_out = o_out * (tanh(inp_tensor=out))
            h_t = o_out * tanh(h_out)
            return h_out, h_t
        else:
            i_out = x.matmul(self._parameters["w_ii"].value) + h_prev.matmul(self._parameters["w_hi"].value)
            f_out = x.matmul(self._parameters["w_if"].value) + h_prev.matmul(self._parameters["w_hf"].value)
            g_out = x.matmul(self._parameters["w_ig"].value) + h_prev.matmul(self._parameters["w_hg"].value)
            o_out = x.matmul(self._parameters["w_io"].value) + h_prev.matmul(self._parameters["w_ho"].value)
            out = (f_out * c_prev) + (i_out * g_out)
            h_out = o_out * (tanh(inp_tensor=out))
            h_t = o_out * tanh(h_out)
            return h_out, h_t


class Dropout(Module):
    """
    it helps to less overfit the model

    Args:
        Module (_type_): _description_
        self.dropout = float
        self.mask = np.ndarray
    """

    def __init__(self, dropout: float) -> None:
        self.dropout = dropout
        self.mask = None

    def forward(self, x: Tensor):
        self.mask = Tensor(data=(np.random.rand(*x.shape()) > self.dropout) /
                           (1.0 - self.dropout), dtype=np.float32, requires_grad=True)
        return self.mask * x
