from __future__ import annotations
import numpy as np
from .utils import assert_2d, assert_1d

def least_squares_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min_w ||Xw - y||^2. Returns w."""
    X = assert_2d(X, "X")
    y = assert_1d(y, "y")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X rows must match y length.")

    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return w
    return np.linalg.inv(XtX) @ X.T @ y

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression: min_w ||Xw - y||^2 + lam||w||^2."""
    X = assert_2d(X, "X")
    y = assert_1d(y, "y")
    d = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T @ y
