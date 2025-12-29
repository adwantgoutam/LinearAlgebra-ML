import numpy as np
import matplotlib.pyplot as plt

from src.least_squares import least_squares_fit

def make_data(n=200, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=n)
    y = 3.0 * x - 2.0 + rng.normal(0, noise, size=n)
    return x, y

def design_matrix(x):
    # X = [x, 1] to learn slope + bias
    return np.column_stack([x, np.ones_like(x)])

def mse(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))

def gradient_descent(X, y, lr=0.05, steps=2000):
    # TODO: implement GD for min_w ||Xw - y||^2
    # Hint: gradient = (2/n) * X^T (Xw - y)
    n, d = X.shape
    w = np.zeros(d)
    history = []
    for t in range(steps):
        y_hat = X @ w
        loss = mse(y_hat, y)
        history.append(loss)

        # TODO: compute grad and update w
        grad = (2.0 / n) * (X.T @ (y_hat - y))
        w = w - lr * grad

    return w, np.array(history)

def main():
    x, y = make_data()
    X = design_matrix(x)

    # Closed form
    w_ls = least_squares_fit(X, y)

    # Gradient descent
    w_gd, loss_hist = gradient_descent(X, y)

    print("True:  y = 3x - 2")
    print("Least squares w = [slope, bias] =", w_ls)
    print("Gradient desc  w = [slope, bias] =", w_gd)

    # Plot fit
    xx = np.linspace(x.min(), x.max(), 200)
    Xx = design_matrix(xx)
    plt.figure()
    plt.scatter(x, y, alpha=0.3, label="data")
    plt.plot(xx, Xx @ w_ls, label="least squares")
    plt.plot(xx, Xx @ w_gd, label="gradient descent")
    plt.legend()
    plt.title("Linear Regression: Least Squares vs Gradient Descent")
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(loss_hist)
    plt.title("Gradient Descent Loss (MSE)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

if __name__ == "__main__":
    main()
