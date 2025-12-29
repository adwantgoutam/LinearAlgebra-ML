from __future__ import annotations
import numpy as np

def pca_fit(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA using covariance eigen-decomposition.
    Returns (X_centered, components, explained_variance)
    components shape: (d, k) columns are principal directions.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)  # symmetric cov
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    components = vecs[:, :k]
    return Xc, components, vals[:k]

def pca_transform(X_centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    return X_centered @ components