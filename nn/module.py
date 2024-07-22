from autodiff.diff import Tensor
import os


class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = None

    def __repr__(self):
        return f'Parameter(value={self.value}, grad={self.grad})'


class Module:
    def __init__(self):
        self.training = True
        self._modules = {}
        self._parameters = {}

    def forward(self, *inputs):
        # will be override in every neural network class defined in nn module
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        # no training required , only for evaluation or predicting output from the model
        self.training = False
        for module in self._modules.values():
            module.eval()

    def add_module(self, name, module):
        """Add a child module to the current module."""
        self._modules[name] = module

    def add_parameter(self, name, value):
        self._parameters[name] = Parameter(value)


class Sequential(Module):
    """
        A general sequential class which will be used for any kind of layer that will be 
        defined in nn module
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()

    def __repr__(self) -> str:
        strg = ""
        for layer in self.layers:
            strg += repr(layer)+",\n"
        return strg
