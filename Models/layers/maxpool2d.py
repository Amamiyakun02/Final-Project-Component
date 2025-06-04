class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        x: numpy array with shape (batch_size, channels, height, width)
        """
        self.input = x
        batch_size, channels, height, width = x.shape
        k = self.kernel_size
        s = self.stride

        out_height = (height - k) // s + 1
        out_width = (width - k) // s + 1
        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.argmax = np.zeros_like(self.output, dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * s
                        w_start = j * s
                        window = x[b, c, h_start:h_start+k, w_start:w_start+k]
                        self.output[b, c, i, j] = np.max(window)
                        self.argmax[b, c, i, j] = np.argmax(window)
        return self.output

    def backward(self, grad_output):
        """
        grad_output: numpy array with shape same as self.output
        """
        batch_size, channels, height, width = self.input.shape
        k = self.kernel_size
        s = self.stride
        grad_input = np.zeros_like(self.input)

        out_height, out_width = grad_output.shape[2], grad_output.shape[3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * s
                        w_start = j * s
                        window = self.input[b, c, h_start:h_start+k, w_start:w_start+k]
                        max_idx = self.argmax[b, c, i, j]
                        max_pos = np.unravel_index(max_idx, window.shape)
                        grad_input[b, c, h_start:h_start+k, w_start:w_start+k][max_pos] += grad_output[b, c, i, j]
        return grad_input
