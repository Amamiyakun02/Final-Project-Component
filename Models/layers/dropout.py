class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True  # Default mode

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            return x * self.mask / (1.0 - self.dropout_rate)  # Scale to keep expectation
        else:
            return x  # No dropout during evaluation

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask / (1.0 - self.dropout_rate)
        else:
            return grad_output
