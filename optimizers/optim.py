import initializes.random_init as rinit


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr=lr)
        self.lr = lr
        self.parameters = None

    def step(self, parameters):
        self.parameters = parameters
        for param in self.parameters:
            if param.grad is not None:
                param.value.data -= self.lr * param.grad
            else:
                param.grad = rinit.Initializer().lecun_uniform(shape=param.value.shape(),
                                                               n_in=param.value.shape()[0], requires_grad=True).data
                param.value.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
