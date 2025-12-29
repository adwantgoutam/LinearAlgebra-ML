from __future__ import annotations
import numpy as np

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes {A.shape} and {B.shape}")
    return A @ B

def transpose(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float).T

def is_square(A: np.ndarray) -> bool:
    A = np.asarray(A)
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def inverse(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if not is_square(A):
        raise ValueError("Inverse requires a square matrix.")
    return np.linalg.inv(A)