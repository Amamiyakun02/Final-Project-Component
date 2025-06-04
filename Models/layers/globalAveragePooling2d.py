class GlobalAveragePooling2D:
    def forward(self, x):
        self.input = x  # shape: (B, C, H, W)
        return np.mean(x, axis=(2, 3))  # shape: (B, C)

    def backward(self, grad_output):
        B, C, H, W = self.input.shape
        grad_input = grad_output[:, :, None, None] * np.ones((B, C, H, W)) / (H * W)
        return grad_input
