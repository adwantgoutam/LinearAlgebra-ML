import numpy as np
import matplotlib.pyplot as plt
from src.eigen_pca import pca_fit, pca_transform

np.random.seed(0)
# Elongated cloud
X = np.random.randn(300, 2) @ np.array([[3, 0],[1, 0.5]])

Xc, comps, var = pca_fit(X, k=1)
Z = pca_transform(Xc, comps)        # (n,1)
X_recon = Z @ comps.T + X.mean(axis=0, keepdims=True)

plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.3, label="original")
plt.scatter(X_recon[:,0], X_recon[:,1], alpha=0.3, label="rank-1 recon")
plt.legend()
plt.title("PCA: original vs rank-1 reconstruction")
plt.show()

print("Top explained variance =", var[0])
print("Principal direction =", comps[:,0])
