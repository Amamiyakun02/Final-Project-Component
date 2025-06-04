class ReLU:
    def __init__(self):
        self.mask = None  # simpan posisi > 0

    def forward(self, x):
        """
        x: input (batch, height, width, channels)
        """
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        """
        grad_output: gradien dari layer selanjutnya
        """
        return grad_output * self.mask
