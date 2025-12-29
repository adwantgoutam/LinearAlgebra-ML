# Project 06 — Sinusoidal Positional Encoding (Transformer Math)

## Goal
Implement the original **sinusoidal positional encoding** (Vaswani et al.) and visualize it.

This teaches:
- why we add position information to embeddings
- how a deterministic basis of sin/cos encodes positions
- how dot products reflect relative position patterns

## Success Criteria
- Generate PE matrix of shape (T, d)
- Visualize as heatmap
- Show similarity between positions using dot products

## Stretch Goals
- Compare adding PE vs not adding in attention demo
- Implement learned positional embeddings (table lookup)
