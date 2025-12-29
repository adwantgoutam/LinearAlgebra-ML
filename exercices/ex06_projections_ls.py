import numpy as np
from src.least_squares import least_squares_fit, ridge_fit

# Linear regression as projection / least squares
# y ≈ w0*x + w1 (bias)  -> X is [x, 1]
X = np.array([[1, 1],
              [2, 1],
              [3, 1],
              [4, 1]], dtype=float)
y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1

w = least_squares_fit(X, y)
print("Least squares w =", w)
print("Predictions =", X @ w)

wr = ridge_fit(X, y, lam=1.0)
print("Ridge w (lam=1.0) =", wr)
