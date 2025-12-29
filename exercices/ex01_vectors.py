import numpy as np
from src.vectors import dot, norm, cosine_similarity, angle_degrees, projection

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print("a =", a)
print("b =", b)
print("dot(a,b) =", dot(a,b))
print("||a|| =", norm(a))
print("cosine(a,b) =", cosine_similarity(a,b))
print("angle(a,b) degrees =", angle_degrees(a,b))

u = np.array([3, 1], dtype=float)
v = np.array([1, 0], dtype=float)
print("proj(u on v) =", projection(u, v))
