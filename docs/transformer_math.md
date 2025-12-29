# Transformer / LLM Linear Algebra Cheatsheet

## 1) Self-attention (single head)

Given token representations `X` with shape `(T, d)`:

```text
Q = X Wq
K = X Wk
V = X Wv
```

Then:

```text
scores  = (Q K^T) / sqrt(d)
weights = softmax(scores)          # row-wise
context = weights V
```

Shapes:
- `Q, K, V`: `(T, d)`
- `scores`: `(T, T)`
- `weights`: `(T, T)`
- `context`: `(T, d)`

Interpretation:
- `scores[i, j]` is how much token `i` attends to token `j` (via dot product similarity)
- `weights` rows sum to 1 (probability distribution over tokens)

## 2) Causal masking (decoder attention)

To prevent looking ahead, set future positions to a very negative number before softmax:

```text
scores[i, j] = -∞  for j > i
```

## 3) Positional encoding

Transformers add position information to token embeddings:

```text
X_pos = X + PE
```

Sinusoidal PE creates a deterministic basis across dimensions where positions have structured similarity.

## 4) Low-rank updates (LoRA intuition)

Adapt weights with a low-rank update:

```text
W' = W + A B
```

where `A` and `B` are small and `rank(AB) <= r`.
