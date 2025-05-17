import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        # Simpan input untuk backward pass
        self.x = x

        batch_size, in_c, in_h, in_w = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding

        # Padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        self.x_padded = x_padded
        out_h = (in_h + 2 * p - k) // s + 1
        out_w = (in_w + 2 * p - k) // s + 1
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(in_c):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start, h_end = i * s, i * s + k
                            w_start, w_end = j * s, j * s + k
                            region = x_padded[b, :, h_start:h_end, w_start:w_end]
                            out[b, oc, i, j] = np.sum(region * self.weights[oc]) + self.bias[oc]
        return out

    def backward(self, grad_output, learning_rate):
        batch_size, in_c, in_h, in_w = self.x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        _, _, out_h, out_w = grad_output.shape

        grad_x_padded = np.zeros_like(self.x_padded)
        grad_w = np.zeros_like(self.weights)
        grad_b = np.zeros_like(self.bias)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i * s, i * s + k
                        w_start, w_end = j * s, j * s + k
                        region = self.x_padded[b, :, h_start:h_end, w_start:w_end]

                        grad_w[oc] += grad_output[b, oc, i, j] * region
                        grad_x_padded[b, :, h_start:h_end, w_start:w_end] += grad_output[b, oc, i, j] + self.weights[oc]
                grad_b[oc] += np.sum(grad_output[b, oc])

        if p > 0:
            grad_input = grad_x_padded[:, : p:-p, p:-p]
        else:
            grad_input = grad_x_padded

        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

        return grad_input