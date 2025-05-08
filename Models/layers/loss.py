import numpy as np

class SoftmaxCrossEntropyLoss:
    def forward(self, logits, labels):
        self.labels = labels
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        log_probs = -np.log(self.probs[range(len(labels)), labels])
        return np.mean(log_probs)

    def backward(self):
        grad = self.probs
        grad[range(len(self.labels)), self.labels] -= 1
        return grad / len(self.labels)

