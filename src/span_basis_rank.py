from __future__ import annotations
import numpy as np
from .utils import assert_2d, assert_1d

def rank(A: np.ndarray, tol: float = 1e-10) -> int:
    A = assert_2d(A, "A")
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

def is_linearly_independent(vectors: list[np.ndarray], tol: float = 1e-10) -> bool:
    if len(vectors) == 0:
        return True
    V = np.column_stack([assert_1d(v, f"v{i}") for i, v in enumerate(vectors)])
    return rank(V, tol=tol) == V.shape[1]

def in_span(v: np.ndarray, basis: list[np.ndarray], tol: float = 1e-8) -> bool:
    v = assert_1d(v, "v")
    if len(basis) == 0:
        return np.allclose(v, 0, atol=tol)
    B = np.column_stack([assert_1d(b, f"b{i}") for i, b in enumerate(basis)])
    x, *_ = np.linalg.lstsq(B, v, rcond=None)
    residual = np.linalg.norm(B @ x - v)
    return float(residual) < tol

def basis_from_columns(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for the column space of A via SVD."""
    A = assert_2d(A, "A")
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(S > tol))
    return U[:, :r]
