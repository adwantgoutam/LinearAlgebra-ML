import numpy as np
import matplotlib.pyplot as plt

from src.eigen_pca import pca_fit, pca_transform

def make_cloud(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)) @ np.array([[4.0, 0.0], [1.5, 0.5]])
    return X

def main():
    X = make_cloud()
    Xc, comps, var = pca_fit(X, k=1)
    Z = pca_transform(Xc, comps)  # (n, 1)

    # reconstruct back to 2D
    X_recon = Z @ comps.T + X.mean(axis=0, keepdims=True)

    print("Explained variance (top-1):", var[0])
    print("Principal direction:", comps[:, 0])

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], alpha=0.25, label="original")
    plt.scatter(X_recon[:, 0], X_recon[:, 1], alpha=0.25, label="reconstruction")
    plt.legend()
    plt.title("PCA Compression: 2D → 1D → 2D")
    plt.show()

    err = np.linalg.norm(X - X_recon) / np.linalg.norm(X)
    print("Relative reconstruction error:", float(err))

if __name__ == "__main__":
    main()
