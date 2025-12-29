from __future__ import annotations
import numpy as np
from .utils import assert_2d

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = assert_2d(A, "A")
    B = assert_2d(B, "B")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes {A.shape} and {B.shape}")
    return A @ B

def transpose(A: np.ndarray) -> np.ndarray:
    A = assert_2d(A, "A")
    return A.T

def identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)

def is_square(A: np.ndarray) -> bool:
    A = np.asarray(A)
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def inverse(A: np.ndarray) -> np.ndarray:
    A = assert_2d(A, "A")
    if not is_square(A):
        raise ValueError("Inverse requires a square matrix.")
    return np.linalg.inv(A)

def column_space_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Numerical rank based on singular values."""
    A = assert_2d(A, "A")
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))
