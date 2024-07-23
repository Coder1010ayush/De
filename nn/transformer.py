"""
    end to end implementation of Transformer model architecture 
    As mentioned in Attention - all you need paper    
    Paper link is here => https://arxiv.org/pdf/1706.03762
"""
import os
import numpy as np
import sys
from initializes.random_init import Initializer as rinit
from nn.module import Module, Sequential, Parameter
from nn.networkx import Linear
from nn.networkx import Embedding
from optimizers.optim import SGD
from nn.functonal import Relu, Sigmoid, Softmax
from autodiff.diff import Tensor


class InputDataEmbedding(Module):

    """
        this class will change the input data (sequence) into the embedding vector 
        of [vocab_size , d_model] as defined in the paper.
        Args:
            d_model : dimension of the embedding
            vocab_size : total number of words in the corpus dataset(unique)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Embedding(vocab_size=self.vocab_size, dim=self.d_model)

    def forward(self, x):
        out = self.embedding(x) * rinit().constants(shape=(1, 1), val=self.d_model,
                                                    dtype=np.float32, requires_grad=True).sqrt()
        return out


class PositionalEncoding(Module):
    """_summary_
    this class will determine the position encoding of a given embedding as defined in paper.
    there is no learnable parameters are involved in this step.
    Args:
        Module (_type_): _description_
        seq_length : length of given sequence in forward pass
        d_model : dimension of encoding will be generated
        dropout : may be including for less overfit 
    """

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.dropout = dropout  # todo for implementation in networkx.py
        self.d_model = d_model
        self.seq_length = seq_length

        # defining the positions
        self.position = rinit().arange(n1=0, n2=seq_length, n3=1, dtype=np.float32, requires_grad=False)
        div_term = (rinit().arange(0, self.d_model, 2) * -
                    ((rinit().constants(shape=(), val=10000.0, dtype=np.float32)) / self.d_model).log())
        self.encoding = rinit().zeros(shape=(self.seq_length, self.d_model), dtype=np.float32, requires_grad=False)

        out1 = self.position * div_term
        self.encoding[:, 0::2] = out1.sin()
        out2 = self.position * div_term
        self.encoding[:, 1::2] = out1.cos()

    def forward(self, x: Tensor):
        out = x + self.encoding[:, :x.shape()[1], :]
        return out


class Attention:

    """
        this class implement Attention layer 
        Args:
            d_model : dimension of output and input seq
    """

    def __init__(self, d_model) -> None:
        self.d_model = d_model
        self.query_mat = Linear(in_features=self.d_model, out_features=self.d_model, bias_option=True)
        self.key_mat = Linear(in_features=self.d_model, out_features=self.d_model, bias_option=True)
        self.value_mat = Linear(in_features=self.d_model, out_features=self.d_model, bias_option=True)

    def forward(self, key, query, value):
        q_out: Tensor = self.query_mat(query)
        k_out: Tensor = self.key_mat(key)
        v_out: Tensor = self.value_mat(value)

        scores = q_out.matmul(other=k_out.transpose())
        score_weight = Softmax().forward(inp_tensor=scores, dim=-1)
        final_score = score_weight.matmul(other=v_out)
        return final_score
