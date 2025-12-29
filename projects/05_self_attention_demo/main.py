import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q,K,V: (T, d). Returns (context, weights)."""
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)               # (T, T)
    if mask is not None:
        scores = scores + mask                    # add -inf on masked positions
    W = softmax(scores, axis=1)                   # row-wise softmax
    context = W @ V                               # (T, d)
    return context, W

def causal_mask(T):
    # mask future positions: upper triangle (excluding diagonal)
    m = np.triu(np.ones((T, T), dtype=float), k=1)
    return -1e9 * m

def main():
    np.random.seed(0)

    # Pretend we have T tokens, each has embedding dimension d
    T, d = 6, 8
    X = np.random.normal(size=(T, d))

    # Linear projections to get Q, K, V (like Wq, Wk, Wv)
    Wq = np.random.normal(scale=0.5, size=(d, d))
    Wk = np.random.normal(scale=0.5, size=(d, d))
    Wv = np.random.normal(scale=0.5, size=(d, d))

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Unmasked attention (bidirectional)
    context, W = scaled_dot_product_attention(Q, K, V, mask=None)

    print("X shape:", X.shape)
    print("Q,K,V shape:", Q.shape)
    print("Attention weights W shape:", W.shape)
    print("Context shape:", context.shape)
    print("Row sums (should be 1):", W.sum(axis=1))

    # Visualize attention weights
    plt.figure()
    plt.imshow(W, aspect="auto")
    plt.colorbar()
    plt.title("Self-Attention Weights (Unmasked)")
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.show()

    # Causal attention (decoder-style)
    mask = causal_mask(T)
    context_causal, Wc = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("\nCausal row sums (should be 1):", Wc.sum(axis=1))

    plt.figure()
    plt.imshow(Wc, aspect="auto")
    plt.colorbar()
    plt.title("Self-Attention Weights (Causal Masked)")
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.show()

if __name__ == "__main__":
    main()
