from __future__ import annotations
import numpy as np

def project_onto_subspace(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the column space of B.
    B: (d, k) matrix with k basis vectors as columns (not necessarily orthonormal).
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float)
    # Projection: P = B(B^T B)^-1 B^T
    BtB = B.T @ B
    if np.linalg.matrix_rank(BtB) < BtB.shape[0]:
        raise ValueError("B columns are dependent; BtB not invertible.")
    P = B @ np.linalg.inv(BtB) @ B.T
    return P @ v