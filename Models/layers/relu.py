import numpy as np

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
