# Project 04 — Tiny Dense Layer (Neural Network) With Vectorization

## Goal
Implement a tiny neural network forward pass using pure NumPy:

- Dense layer: `Z = XW + b`
- Activation: ReLU
- Dense layer 2: `Y = A W2 + b2`
- Output: softmax probabilities

Connects to:
- matrix multiplication as batched dot products
- shape thinking (batch, features, hidden units)
- how real frameworks (PyTorch/TF) work under the hood

## Success Criteria
- Print shapes at each stage
- Output probabilities sum to 1 per sample

## Stretch Goals
- Add cross-entropy loss for fake labels
- Implement one step of gradient descent (optional)
