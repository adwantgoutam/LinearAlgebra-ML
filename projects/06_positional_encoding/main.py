import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(T: int, d: int) -> np.ndarray:
    pe = np.zeros((T, d), dtype=float)
    position = np.arange(T)[:, None]  # (T,1)
    div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def cosine_sim_matrix(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # cosine similarity between rows of A
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    An = A / norms
    return An @ An.T

def main():
    T, d = 50, 32
    pe = sinusoidal_positional_encoding(T, d)

    print("PE shape:", pe.shape)

    plt.figure()
    plt.imshow(pe, aspect="auto")
    plt.colorbar()
    plt.title("Sinusoidal Positional Encoding (T=50, d=32)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.show()

    S = cosine_sim_matrix(pe)
    plt.figure()
    plt.imshow(S, aspect="auto")
    plt.colorbar()
    plt.title("Cosine Similarity Between Positions (PE rows)")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.show()

if __name__ == "__main__":
    main()
