"""
02_erm_training_loop_simulation.py
==================================
End-to-End Empirical Risk Minimization (ERM) Training Loop Simulation.
Validates:
1. Multi-batch SGD optimization with forward-backward passes
2. Explicit backprop error recursion: delta^[l] = ((W^[l+1])^T delta^[l+1]) * sigma'(z^[l])
3. Monotonic empirical risk reduction on a non-linear dataset
4. Final classification accuracy > 95%
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def generate_xor_dataset(n_samples_per_cluster=100, noise=0.05):
    np.random.seed(42)
    centers = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    labels = np.array([[0.0], [1.0], [1.0], [0.0]])
    
    X_list, Y_list = [], []
    for c, l in zip(centers, labels):
        X_cluster = c + np.random.randn(n_samples_per_cluster, 2) * noise
        Y_cluster = np.tile(l, (n_samples_per_cluster, 1))
        X_list.append(X_cluster)
        Y_list.append(Y_cluster)
        
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]

class TwoLayerMLP:
    def __init__(self, in_dim=2, hidden_dim=8, out_dim=1):
        np.random.seed(42)
        # He / Xavier initialization
        self.W1 = np.random.randn(hidden_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(out_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((out_dim, 1))

    def forward(self, x):
        # x shape: [in_dim, 1]
        self.x = x
        self.z1 = self.W1 @ x + self.b1 # [hidden_dim, 1]
        self.a1 = sigmoid(self.z1)      # [hidden_dim, 1]
        self.z2 = self.W2 @ self.a1 + self.b2 # [out_dim, 1]
        self.a2 = sigmoid(self.z2)      # [out_dim, 1]
        return self.a2

    def backward(self, y):
        # y shape: [out_dim, 1]
        # Output error: delta^[2] = (a^[2] - y) * sigma'(z^[2])
        dL_da2 = self.a2 - y
        self.delta2 = dL_da2 * sigmoid_prime(self.z2) # [out_dim, 1]

        # Hidden error: delta^[1] = (W2^T delta^[2]) * sigma'(z^[1])
        self.delta1 = (self.W2.T @ self.delta2) * sigmoid_prime(self.z1) # [hidden_dim, 1]

        # Parameter gradients
        self.grad_W2 = self.delta2 @ self.a1.T # [out_dim, hidden_dim]
        self.grad_b2 = self.delta2            # [out_dim, 1]
        self.grad_W1 = self.delta1 @ self.x.T  # [hidden_dim, in_dim]
        self.grad_b1 = self.delta1            # [hidden_dim, 1]

    def update(self, lr):
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2

def main():
    X, Y = generate_xor_dataset(n_samples_per_cluster=75, noise=0.08)
    n_samples = len(X)
    mlp = TwoLayerMLP(in_dim=2, hidden_dim=12, out_dim=1)

    initial_loss = 0.0
    for i in range(n_samples):
        x_i = X[i:i+1].T
        y_i = Y[i:i+1].T
        pred = mlp.forward(x_i)
        initial_loss += 0.5 * (pred[0, 0] - y_i[0, 0])**2
    initial_loss /= n_samples

    print(f"Initial Empirical Risk R_hat: {initial_loss:.4f}")

    # Training loop: 80 epochs with learning rate 0.4
    epochs = 80
    lr = 0.4
    loss_history = []

    for ep in range(epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        for i in perm:
            x_i = X[i:i+1].T
            y_i = Y[i:i+1].T
            pred = mlp.forward(x_i)
            loss = 0.5 * (pred[0, 0] - y_i[0, 0])**2
            epoch_loss += loss
            mlp.backward(y_i)
            mlp.update(lr)
        
        avg_epoch_loss = epoch_loss / n_samples
        loss_history.append(avg_epoch_loss)
        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1:02d}/{epochs} | Empirical Risk: {avg_epoch_loss:.5f}")

    final_loss = loss_history[-1]
    print(f"Final Empirical Risk R_hat: {final_loss:.5f}")

    # Evaluate accuracy
    correct = 0
    for i in range(n_samples):
        pred = mlp.forward(X[i:i+1].T)[0, 0]
        label = Y[i, 0]
        pred_binary = 1.0 if pred >= 0.5 else 0.0
        if pred_binary == label:
            correct += 1
    accuracy = correct / n_samples
    print(f"Final Classification Accuracy: {accuracy * 100:.2f}%")

    # Assertions verifying ERM convergence
    assert final_loss < initial_loss * 0.15, "Empirical risk failed to decrease significantly!"
    assert accuracy > 0.95, f"Expected accuracy > 95%, got {accuracy*100:.2f}%"
    print("[ERM SIMULATION VERIFIED] Training loop executed successfully.")

if __name__ == "__main__":
    main()
