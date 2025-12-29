import numpy as np
import matplotlib.pyplot as plt

# Linear transform example: rotation + scaling
X = np.array([[1,0],[0,1],[1,1],[-1,1]], dtype=float)  # points
theta = np.deg2rad(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]], dtype=float)
S = np.array([[2,0],[0,0.5]], dtype=float)
A = R @ S

Y = X @ A  # transform

plt.figure()
plt.scatter(X[:,0], X[:,1], label="original")
plt.scatter(Y[:,0], Y[:,1], label="transformed")
plt.axhline(0); plt.axvline(0)
plt.legend()
plt.title("Original points and transformed points")
plt.show()
