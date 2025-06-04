class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.input = x  # shape: (B, D)
        return x @ self.W + self.b  # shape: (B, K)

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output  # shape: (D, K)
        self.grad_b = np.sum(grad_output, axis=0)  # shape: (K,)
        grad_input = grad_output @ self.W.T  # shape: (B, D)
        return grad_input
