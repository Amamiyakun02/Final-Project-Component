class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters  # list of params (weights & biases)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if hasattr(param, 'grad'):
                param -= self.lr * param.grad
