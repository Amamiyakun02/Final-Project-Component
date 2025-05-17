import numpy as np

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        self.mask = self.mask * grad > 0
        return grad * self.mask