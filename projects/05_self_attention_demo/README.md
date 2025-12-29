# Project 05 — Self-Attention (Core of Transformers / LLMs)

## Goal
Implement **scaled dot-product self-attention** with pure NumPy:

```text
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
```

This project is *the* linear algebra heart of Transformers and LLMs.

## What you will practice
- dot products as similarity
- matrix multiplication as batched dot products
- softmax as normalization
- shape discipline: (tokens × dim) and (tokens × tokens)

## Success Criteria
- Compute attention weights (tokens × tokens)
- Show they row-sum to 1
- Produce context vectors (tokens × dim)

## Stretch Goals
- Add a causal mask (prevent looking ahead)
- Add multiple heads (split dim into heads)
- Visualize attention heatmap
