from __future__ import annotations
import numpy as np

def assert_1d(v: np.ndarray, name: str = "v") -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D vector, got shape {v.shape}")
    return v

def assert_2d(A: np.ndarray, name: str = "A") -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D matrix, got shape {A.shape}")
    return A

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = assert_1d(v)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def pretty(x: np.ndarray, decimals: int = 4) -> str:
    return np.array2string(np.asarray(x, dtype=float), precision=decimals, suppress_small=True)
