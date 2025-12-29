# Learning Flow: Linear Algebra → ML / Neural Networks

This repo is organized as a **progressive path**. Each concept builds on the previous one.

```text
Scalars → Vectors → Matrices
           │          │
           ├─ Dot product (similarity, neuron pre-activation)
           │
           ├─ Span / Basis / Rank (redundancy, feature space)
           │
           ├─ Projections → Least Squares (linear regression geometry)
           │
           ├─ Eigen / PCA (new basis that captures variance)
           │
           └─ SVD / Low-rank (compression, embeddings, efficient layers)
```

## How this maps to a Neural Network layer

A dense layer is:

```text
Z = XW + b
A = activation(Z)
```

- **X** is a matrix: (batch_size × input_dim)
- **W** is a matrix: (input_dim × output_dim)
- **b** is a vector: (output_dim,)
- **XW** is matrix multiplication (many dot products)
- each output unit is a dot product between one input row and one column of W

### Visual: a single neuron as a dot product

```text
x = [x1, x2, x3]
w = [w1, w2, w3]

z = w·x + b = (w1*x1 + w2*x2 + w3*x3) + b
a = f(z)
```

## How to practice here
- Start in `exercises/` in order.
- Read the matching doc sections in `docs/`.
- Use `tests/` to confirm you implemented correctly.
