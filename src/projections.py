from __future__ import annotations
import numpy as np
from .utils import assert_1d, assert_2d

def project_onto_subspace(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project v onto the column space of B (columns not necessarily orthonormal)."""
    v = assert_1d(v, "v")
    B = assert_2d(B, "B")

    BtB = B.T @ B
    if np.linalg.matrix_rank(BtB) < BtB.shape[0]:
        raise ValueError("Columns of B are linearly dependent; (B^T B) not invertible.")
    P = B @ np.linalg.inv(BtB) @ B.T
    return P @ v

def project_onto_orthonormal_subspace(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """If Q has orthonormal columns, projection simplifies to Q Q^T v."""
    v = assert_1d(v, "v")
    Q = assert_2d(Q, "Q")
    return Q @ (Q.T @ v)
