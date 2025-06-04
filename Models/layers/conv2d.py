import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        in_channels: jumlah channel input (misal: 3 untuk RGB)
        out_channels: jumlah filter
        kernel_size: ukuran kernel (biasanya 3)
        stride: pergeseran filter
        padding: jumlah padding di sekeliling input
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight: (out_channels, in_channels, kernel_size, kernel_size)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)

        # Gradien
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        x: input dengan shape (batch_size, height, width, in_channels)
        output: (batch_size, new_height, new_width, out_channels)
        """
        self.x = x
        batch_size, h_in, w_in, _ = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Padding
        x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        self.x_padded = x_padded

        h_out = (h_in + 2*p - k) // s + 1
        w_out = (w_in + 2*p - k) // s + 1
        out = np.zeros((batch_size, h_out, w_out, self.out_channels))

        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for f in range(self.out_channels):
                        h_start = i * s
                        w_start = j * s
                        region = x_padded[b, h_start:h_start+k, w_start:w_start+k, :]
                        out[b, i, j, f] = np.sum(region * self.W[f]) + self.b[f]
        return out

    def backward(self, grad_output, lr):
        """
        grad_output: gradien dari layer berikutnya, shape (batch, h_out, w_out, out_channels)
        """
        batch_size, h_out, w_out, _ = grad_output.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        self.dW.fill(0)
        self.db.fill(0)
        dx_padded = np.zeros_like(self.x_padded)

        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for f in range(self.out_channels):
                        h_start = i * s
                        w_start = j * s
                        region = self.x_padded[b, h_start:h_start+k, w_start:w_start+k, :]

                        self.dW[f] += region * grad_output[b, i, j, f]
                        self.db[f] += grad_output[b, i, j, f]
                        dx_padded[b, h_start:h_start+k, w_start:w_start+k, :] += self.W[f] * grad_output[b, i, j, f]

        # Update weight & bias
        self.W -= lr * self.dW
        self.b -= lr * self.db

        # Remove padding
        if p > 0:
            dx = dx_padded[:, p:-p, p:-p, :]
        else:
            dx = dx_padded
        return dx
