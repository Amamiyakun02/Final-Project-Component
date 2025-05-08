import numpy as np

class FullyConnected:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias
