# Project 07 — Low-Rank Updates (LoRA Intuition)

## Goal
Show how a weight matrix can be adapted using a **low-rank update**:

```text
W' = W + ΔW
ΔW = A B   where A is (d_in × r), B is (r × d_out), and r is small
```

This is the core linear algebra idea behind **LoRA-style** fine-tuning:
- keep base weights W frozen
- learn small matrices A and B

## Success Criteria
- Create random W and a target W*
- Fit a low-rank ΔW to reduce error
- Show error improves as rank r increases

## Stretch Goals
- Use gradient descent on A and B (optional)
- Compare full-rank vs low-rank parameter counts
