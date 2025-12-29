# Visual Explanations (ML intuition)

## Dot product = alignment / similarity

```text
a·b = |a||b|cos(θ)

θ small  → cos(θ) ≈ 1 → strong alignment
θ = 90°  → cos(θ) = 0 → orthogonal (no linear relation)
θ > 90°  → cos(θ) < 0 → opposite directions
```

**ML usage**
- cosine similarity between embeddings
- attention scores (Transformers use dot products heavily)

---

## Matrix multiplication = “many dot products”

```text
(XW)[i, j] = dot( X[i, :], W[:, j] )
```

Think:
- one row of X = one sample (feature vector)
- one column of W = weights of one neuron
- output entry = that neuron's pre-activation for that sample

---

## Span / Basis / Rank = “how many useful directions exist”

```text
If v3 = a*v1 + b*v2, then v3 adds no new direction → redundancy
Rank = number of independent directions
```

**ML usage**
- rank tells effective dimensionality of your data/features
- PCA chooses a new basis to keep the top variance directions

---

## Least Squares = projection

Linear regression is:

```text
choose w that minimizes ||Xw - y||^2
```

Geometrically:
- y is projected onto the column space of X
- Xw is the closest point (prediction) to y within that space
