
class Optimizer:
    def __init__(self, *inputs) -> None:
        pass

    def step(self):
        return NotImplementedError()

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
