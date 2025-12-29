from __future__ import annotations
import numpy as np

def least_squares_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve min_w ||Xw - y||^2 via normal equation (or lstsq if preferred).
    X: (n, d), y: (n,)
    Returns w: (d,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    # Normal equation: w = (X^T X)^-1 X^T y
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        # fall back to least squares if not invertible
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return w
    return np.linalg.inv(XtX) @ X.T @ y