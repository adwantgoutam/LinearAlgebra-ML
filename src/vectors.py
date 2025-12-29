from __future__ import annotations
import numpy as np
from .utils import assert_1d, normalize

def dot(a: np.ndarray, b: np.ndarray) -> float:
    a = assert_1d(a, "a")
    b = assert_1d(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same shape, got {a.shape} and {b.shape}")
    return float(np.sum(a * b))

def norm(v: np.ndarray) -> float:
    v = assert_1d(v, "v")
    return float(np.sqrt(dot(v, v)))

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity = cos(theta) = (a·b)/(|a||b|)."""
    a = assert_1d(a, "a")
    b = assert_1d(b, "b")
    denom = norm(a) * norm(b)
    if denom < eps:
        raise ValueError("Cannot compute cosine similarity with near-zero norm.")
    return dot(a, b) / denom

def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between vectors in degrees."""
    cos = cosine_similarity(a, b)
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def projection(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project vector u onto vector v (not necessarily unit)."""
    u = assert_1d(u, "u")
    v = assert_1d(v, "v")
    denom = dot(v, v)
    if abs(denom) < 1e-12:
        raise ValueError("Cannot project onto near-zero vector.")
    return (dot(u, v) / denom) * v

def unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector (direction) of v."""
    return normalize(v)
