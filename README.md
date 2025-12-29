# Linear Algebra for ML (Practice Repo)

A **hands-on** linear algebra repo focused on the pieces you actually use in **Machine Learning / Deep Learning**:
dot products, matrix multiplication, span/basis/rank, projections (least squares), PCA, and SVD.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the demo:

```bash
python scripts/quick_demo.py
```

Run exercises (in order):

```bash
python exercises/ex01_vectors.py
python exercises/ex02_matrices.py
python exercises/ex03_dot_cosine.py
python exercises/ex04_transforms.py
python exercises/ex05_span_rank.py
python exercises/ex06_projections_ls.py
python exercises/ex07_pca.py
python exercises/ex08_svd.py
```

Run tests:

```bash
pytest -q
```

---

## Learning flow (recommended path)

Read: `docs/learning_flow.md`  
Visuals: `docs/visuals.md`

```text
Vectors → Matrices → Span/Basis/Rank → Projections → Least Squares → PCA → SVD
```

---

## Repo structure

- `src/` – clean implementations (NumPy)
- `exercises/` – runnable scripts for practice
- `notebooks/` – guided notebooks (fill in as you learn)
- `docs/` – visual explanations & concept map
- `tests/` – sanity checks

---

## Neural Networks connection (one dense layer)

```text
Z = XW + b
A = activation(Z)
```

and each element is a dot product:

```text
Z[i, j] = dot( X[i, :], W[:, j] )
```

---

## License
MIT (use freely).

_Generated on 2025-12-29_

---

## Mini projects

Run the mini projects in `projects/` to connect linear algebra to real ML / NN workflows:

```bash
python projects/01_linear_regression_from_scratch/main.py
python projects/02_pca_compression/main.py
python projects/03_embedding_similarity_search/main.py
python projects/04_tiny_dense_layer_nn/main.py
python projects/05_self_attention_demo/main.py
python projects/06_positional_encoding/main.py
python projects/07_low_rank_update_lora_intuition/main.py
```

See `projects/README.md` for details and stretch goals.

