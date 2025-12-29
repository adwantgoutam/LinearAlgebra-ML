from __future__ import annotations
import numpy as np

def dot(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Vectors must have same shape.")
    return float(np.sum(a * b))

def norm(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float).reshape(-1)
    return float(np.sqrt(dot(v, v)))

def angle_cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """cos(theta) between a and b (cosine similarity without mean-centering)."""
    denom = norm(a) * norm(b)
    if denom < eps:
        raise ValueError("Cannot compute angle cosine with near-zero norm.")
    return dot(a, b) / denom

def projection(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project vector u onto vector v."""
    v = np.asarray(v, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    denom = dot(v, v)
    if abs(denom) < 1e-12:
        raise ValueError("Cannot project onto near-zero vector.")
    return (dot(u, v) / denom) * v