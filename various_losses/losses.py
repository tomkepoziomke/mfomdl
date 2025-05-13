import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def ys_interval(X, a, b, prob, outlier_offset, start, stop, single_side=True):
    ys = X * a + b + np.random.randn(X.shape[0]) * 2
    n = X.shape[0]
    start, stop = max(0, start), min(1, stop)

    rand = None
    if single_side:
        rand = np.random.binomial(1, prob, size=n) * (np.random.randint(0, 2, size=n)) * outlier_offset
    else:
        rand = np.random.binomial(1, prob, size=n) * (np.random.randint(0, 2, size=n) * 2 - 1) * outlier_offset
    ind = np.zeros_like(ys)
    ind[int(n * start) : int(n * stop)] = 1
    rand = rand * ind
    
    return ys + rand 


def mae_loss(params, X, y):
    a, b = params
    return np.mean(np.abs(y - (a*X + b)))

def rmse_loss(params, X, y):
    a, b = params
    return np.sqrt(np.mean(np.power(y - (a * X + b), 2)))

def mape_loss(params, X, y):
    a, b = params
    return np.mean(np.abs(y - (a * X + b)) / (np.abs(y) + 1e-10))

def smape_loss(params, X, y):
    a, b = params
    return 2 * np.mean((np.abs(y - (a * X + b)))/(np.abs(y) + np.abs(a * X + b) + 1e-10))

def main():
    a, b = 5, 3
    n = 1000
    outlier_probability = 0.2
    outlier_offset = 30
    outlier_start, outlier_stop = 0, 1
    outlier_single_side = False

    X = np.linspace(0, 10, n)
    Y = ys_interval(X, a, b, 
                    prob=outlier_probability, 
                    outlier_offset=outlier_offset, 
                    start=outlier_start, 
                    stop=outlier_stop, 
                    single_side=outlier_single_side)

    fs = [mae_loss, rmse_loss, mape_loss, smape_loss]
    aux = [
        lambda args: mae_loss(args, X, Y),
        lambda args: rmse_loss(args, X, Y),
        lambda args: mape_loss(args, X, Y),
        lambda args: smape_loss(args, X, Y)
        ]
    labels = ["MAE", "RMS", "MAPE", "SMAPE"]
    results = [minimize(f, x0=np.zeros(2)).x for f in aux]

    for res, f, label in zip(results, fs, labels):
        err = f(res, X, Y)
        print(f"{label} loss: {err}")

    plt.scatter(X, Y, s=1, marker='.', color='#aaa', zorder=9)
    plt.plot(X, a * X + b, label='Exact')
    for label, result in zip(labels, results):
        a_res, b_res = result
        plt.plot(X, a_res * X + b_res, label=label, zorder=10)

    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()