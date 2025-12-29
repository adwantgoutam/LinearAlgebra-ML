from __future__ import annotations
import numpy as np

def assert_shape(x: np.ndarray, shape: tuple[int, ...], name: str = "array") -> None:
    if x.shape != shape:
        raise ValueError(f"{name} has shape {x.shape}, expected {shape}")

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def pretty(x: np.ndarray, decimals: int = 4) -> str:
    return np.array2string(np.asarray(x, dtype=float), precision=decimals, suppress_small=True)