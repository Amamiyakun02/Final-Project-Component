class SoftmaxCrossEntropyLoss:
    def forward(self, logits, labels):
        # logits: (B, K), labels: (B, K) one-hot
        self.logits = logits
        self.labels = labels

        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        # Cross-entropy loss
        batch_size = logits.shape[0]
        loss = -np.sum(labels * np.log(self.probs + 1e-12)) / batch_size
        return loss

    def backward(self):
        # Derivatif loss terhadap logits
        return (self.probs - self.labels) / self.labels.shape[0]
