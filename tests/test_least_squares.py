import numpy as np
from src.least_squares import least_squares_fit

def test_least_squares_fit_line():
    X = np.array([[1,1],[2,1],[3,1],[4,1]], dtype=float)
    y = np.array([3,5,7,9], dtype=float)  # 2x + 1
    w = least_squares_fit(X, y)
    assert np.allclose(w, np.array([2.0, 1.0]), atol=1e-8)
