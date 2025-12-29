import numpy as np
from src.vectors import cosine_similarity
from src.least_squares import least_squares_fit

print("== Embedding similarity demo ==")
a = np.array([0.8, 0.2, 0.1])
b = np.array([0.79, 0.22, 0.1])
print("cosine(a,b) =", cosine_similarity(a,b))

print("\n== Linear regression demo ==")
X = np.array([[1,1],[2,1],[3,1],[4,1]], dtype=float)
y = np.array([3,5,7,9], dtype=float)
w = least_squares_fit(X, y)
print("w =", w)
