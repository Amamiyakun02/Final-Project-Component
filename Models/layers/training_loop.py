import numpy as np

def train(model, dataloader, loss_fn, epochs=10, learning_rate=0.001):
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            # Forward pass
            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            total_loss += loss

            # Backward pass
            grad = loss_fn.backward()
            model.backward(grad, learning_rate)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
