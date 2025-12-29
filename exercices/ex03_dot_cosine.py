import numpy as np
from src.vectors import cosine_similarity

# Toy "embedding" similarity
king = np.array([0.8, 0.2, 0.1])
queen = np.array([0.79, 0.22, 0.1])
car = np.array([-0.1, 0.4, 0.9])

print("cos(king, queen) =", cosine_similarity(king, queen))
print("cos(king, car)   =", cosine_similarity(king, car))
