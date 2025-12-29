import numpy as np
from src.vectors import dot, norm, cosine_similarity

def test_dot():
    assert dot(np.array([1,2,3]), np.array([4,5,6])) == 32.0

def test_norm():
    assert abs(norm(np.array([3,4])) - 5.0) < 1e-9

def test_cosine_similarity():
    a = np.array([1,0])
    b = np.array([0,1])
    assert abs(cosine_similarity(a,b)) < 1e-9
