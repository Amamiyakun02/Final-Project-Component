class BatchNorm2D:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        num_features: jumlah channel fitur (biasanya sama dengan output channel dari Conv2D)
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Parameter trainable
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Untuk inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        """
        x: input dengan shape (batch, height, width, channels)
        """
        self.x = x
        if training:
            self.mean = x.mean(axis=(0, 1, 2))
            self.var = x.var(axis=(0, 1, 2))

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, grad_output, lr):
        """
        grad_output: gradien dari layer berikutnya, shape (batch, height, width, channels)
        """
        N, H, W, C = grad_output.shape

        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * ((self.var + self.epsilon) ** -1.5), axis=(0, 1, 2))
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.epsilon), axis=(0, 1, 2)) + \
                dvar * np.sum(-2 * (self.x - self.mean), axis=(0, 1, 2)) / (N * H * W)

        dx = dx_norm / np.sqrt(self.var + self.epsilon) + \
             dvar * 2 * (self.x - self.mean) / (N * H * W) + \
             dmean / (N * H * W)

        self.dgamma = np.sum(grad_output * self.x_norm, axis=(0, 1, 2))
        self.dbeta = np.sum(grad_output, axis=(0, 1, 2))

        # Update gamma dan beta
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

        return dx
