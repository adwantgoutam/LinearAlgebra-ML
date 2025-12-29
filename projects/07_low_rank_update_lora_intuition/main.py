import numpy as np
import matplotlib.pyplot as plt

def fit_low_rank_update(W: np.ndarray, W_target: np.ndarray, r: int):
    """Compute best rank-r approximation of (W_target - W) using SVD."""
    D = W_target - W
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vtr = Vt[:r, :]
    Delta_r = Ur @ Sr @ Vtr
    return Delta_r

def main():
    np.random.seed(0)
    d_in, d_out = 64, 64

    # Base weights W and a "target" W* (simulating a new task)
    W = np.random.normal(scale=0.5, size=(d_in, d_out))
    W_target = W + np.random.normal(scale=0.2, size=(d_in, d_out))  # small change

    base_err = np.linalg.norm(W_target - W) / np.linalg.norm(W_target)
    print("Base relative error:", float(base_err))

    ranks = [1, 2, 4, 8, 16, 32, 64]
    errs = []
    for r in ranks:
        Delta = fit_low_rank_update(W, W_target, r=r)
        W_prime = W + Delta
        err = np.linalg.norm(W_target - W_prime) / np.linalg.norm(W_target)
        errs.append(err)
        print(f"r={r:>2}  relative error={float(err):.6f}")

    plt.figure()
    plt.plot(ranks, errs, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("rank r")
    plt.ylabel("relative error after low-rank update")
    plt.title("LoRA Intuition: Low-Rank Update Quality vs Rank")
    plt.show()

if __name__ == "__main__":
    main()
