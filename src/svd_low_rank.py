from __future__ import annotations
import numpy as np
from .utils import assert_2d

def low_rank_approx(A: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approximation using SVD."""
    A = assert_2d(A, "A")
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vtr = Vt[:r, :]
    return Ur @ Sr @ Vtr
