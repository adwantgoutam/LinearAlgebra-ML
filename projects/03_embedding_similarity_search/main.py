import numpy as np
from src.vectors import cosine_similarity

def make_toy_embeddings():
    # A tiny fake embedding space: 3D vectors
    items = {
        "king":  np.array([0.80, 0.20, 0.10]),
        "queen": np.array([0.79, 0.22, 0.10]),
        "man":   np.array([0.70, 0.10, 0.00]),
        "woman": np.array([0.69, 0.12, 0.02]),
        "car":   np.array([-0.10, 0.40, 0.90]),
        "truck": np.array([-0.12, 0.38, 0.92]),
    }
    return items

def top_k(query, items, k=3):
    scores = []
    for name, vec in items.items():
        scores.append((name, cosine_similarity(query, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def main():
    items = make_toy_embeddings()
    query = np.array([0.78, 0.21, 0.10])  # "royalty-ish"
    results = top_k(query, items, k=4)

    print("Query:", query)
    print("Top results by cosine similarity:")
    for name, score in results:
        print(f"  {name:>6}  score={score:.4f}")

if __name__ == "__main__":
    main()
