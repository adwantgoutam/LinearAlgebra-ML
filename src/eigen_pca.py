from __future__ import annotations
import numpy as np
from .utils import assert_2d

def pca_fit(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via covariance eigendecomposition. Returns (X_centered, components, explained_variance)."""
    X = assert_2d(X, "X")
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    comps = vecs[:, :k]
    return Xc, comps, vals[:k]

def pca_transform(X_centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    X_centered = assert_2d(X_centered, "X_centered")
    components = assert_2d(components, "components")
    return X_centered @ components
