import numpy as np

class BatchNorm2D:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Parameter trainable
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))

        # Running estimates
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.training = True

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.x_hat = (x - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma * self.x_hat + self.beta
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_hat + self.beta

        return out
