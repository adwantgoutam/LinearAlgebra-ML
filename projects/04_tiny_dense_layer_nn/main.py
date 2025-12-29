import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def softmax(logits):
    # stable softmax
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def main():
    rng = np.random.default_rng(0)

    # Fake "batch" of 5 samples, each with 4 features
    X = rng.normal(size=(5, 4))

    # Layer 1: 4 -> 6
    W1 = rng.normal(scale=0.5, size=(4, 6))
    b1 = np.zeros(6)

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Layer 2: 6 -> 3 classes
    W2 = rng.normal(scale=0.5, size=(6, 3))
    b2 = np.zeros(3)

    logits = A1 @ W2 + b2
    probs = softmax(logits)

    print("X shape:", X.shape)
    print("W1 shape:", W1.shape, "b1 shape:", b1.shape)
    print("Z1 shape:", Z1.shape, "A1 shape:", A1.shape)
    print("W2 shape:", W2.shape, "b2 shape:", b2.shape)
    print("logits shape:", logits.shape, "probs shape:", probs.shape)
    print("\nProbabilities (each row sums to 1):")
    print(probs)
    print("Row sums:", probs.sum(axis=1))

if __name__ == "__main__":
    main()
