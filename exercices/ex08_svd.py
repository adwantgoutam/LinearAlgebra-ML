import numpy as np
import matplotlib.pyplot as plt
from src.svd_low_rank import low_rank_approx

np.random.seed(0)
A = np.random.randn(30, 30)
A[:, 10:] *= 0.2  # make it more compressible-ish

errs = []
for r in [1,2,5,10,15,20]:
    Ar = low_rank_approx(A, r=r)
    err = np.linalg.norm(A-Ar) / np.linalg.norm(A)
    errs.append((r, err))

print("Relative reconstruction errors:", errs)

plt.figure()
plt.plot([r for r,_ in errs], [e for _,e in errs], marker="o")
plt.xlabel("rank r")
plt.ylabel("relative error")
plt.title("Low-rank approximation error vs rank")
plt.show()
