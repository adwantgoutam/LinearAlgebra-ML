import numpy as np
from src.matrices import matmul, transpose, inverse, column_space_rank

A = np.array([[1, 2],
              [3, 4]], dtype=float)
B = np.array([[2, 0],
              [1, 2]], dtype=float)

print("A=\n", A)
print("B=\n", B)
print("A@B=\n", matmul(A,B))
print("A^T=\n", transpose(A))
print("inv(A)=\n", inverse(A))
print("rank(A) =", column_space_rank(A))
