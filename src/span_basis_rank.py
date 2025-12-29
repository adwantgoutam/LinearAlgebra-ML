from __future__ import annotations
import numpy as np

def rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Numerical rank using SVD."""
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

def is_linearly_independent(vectors: list[np.ndarray], tol: float = 1e-10) -> bool:
    """Check independence by stacking vectors as columns and computing rank."""
    if len(vectors) == 0:
        return True
    V = np.column_stack([np.asarray(v, dtype=float).reshape(-1) for v in vectors])
    return rank(V, tol=tol) == V.shape[1]

def in_span(v: np.ndarray, basis: list[np.ndarray], tol: float = 1e-8) -> bool:
    """Check if v lies in span(basis) via least squares residual."""
    v = np.asarray(v, dtype=float).reshape(-1)
    if len(basis) == 0:
        return np.allclose(v, 0, atol=tol)
    B = np.column_stack([np.asarray(b, dtype=float).reshape(-1) for b in basis])
    x, *_ = np.linalg.lstsq(B, v, rcond=None)
    residual = np.linalg.norm(B @ x - v)
    return residual < tol