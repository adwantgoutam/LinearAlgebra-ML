import numpy as np
from src.span_basis_rank import is_linearly_independent, in_span, rank

v1 = np.array([1,0], dtype=float)
v2 = np.array([0,1], dtype=float)
v3 = np.array([1,1], dtype=float)

print("independent [v1,v2]? ", is_linearly_independent([v1,v2]))
print("is v3 in span(v1,v2)?", in_span(v3, [v1,v2]))

A = np.column_stack([v1,v2,v3])  # 2x3
print("A=\n", A)
print("rank(A) =", rank(A))
