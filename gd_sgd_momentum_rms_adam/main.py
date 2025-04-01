import math
import numpy as np
import matplotlib.pyplot as plt

def gd(df, x_0, n, lr):
    current_x = x_0
    history = [current_x]
    for _ in range(n):
        current_x -= lr * df(current_x)
        history.append(current_x)
    return np.array(history)

def sgd(df, x_0, n, lr, sigma):
    current_x = x_0
    history = [current_x]
    for _ in range(n):
        current_x -= lr * df(current_x) + lr * np.random.normal(loc=0, scale=sigma)
        history.append(current_x)
    return np.array(history)

def momentum(df, x_0, n, gamma, lr):
    current_x = x_0
    current_v = 0
    history = [current_x]
    for _ in range(n):
        current_v = gamma * current_v - lr * df(current_x)
        current_x = current_x + current_v
        history.append(current_x)
    return np.array(history)

def rms_propagation(df, x_0, n, beta, lr, eps):
    current_x = x_0
    current_e_df = df(x_0) ** 2
    history = [current_x]
    for _ in range(n):
        current_x -= lr / np.sqrt(current_e_df + eps) * df(current_x)
        current_e_df = (beta * current_e_df + (1 - beta) * df(current_x) ** 2)
        history.append(current_x)

    return np.array(history)

def adam(df, x_0, n, lr, b1, b2, eps):
    current_x = x_0
    current_m = 0
    current_v = 0
    history = [current_x]

    for _ in range(n):
        current_m = b1 * current_m + (1 - b1) * df(current_x)
        current_v = b2 * current_v + (1 - b2) * (df(current_x)) ** 2
        current_x -= lr / (np.sqrt(current_v) + eps) * current_m
        history.append(current_x)

    return np.array(history)

def multivariate_gd(Df, x_0, n, lr):
    current_x = np.array(x_0)
    history = [current_x.copy()]
    for _ in range(n):
        new_x = current_x - lr * np.array(Df(*current_x))
        history.append(new_x)
        current_x = new_x
    return np.array(history)

def plot_me(hist, f_true, projection='2d'):
    if projection == '2d':
        plt.plot(hist, f_true(hist))
        x_min, x_max = np.min(hist), np.max(hist)
        linspace = np.linspace(x_min, x_max, num=30)
        plt.plot(linspace, f_true(linspace))
        plt.scatter(hist, f_true(hist))
        plt.show()

    elif projection == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x_min, x_max = np.min(hist[:, 0]), np.max(hist[:, 0])
        y_min, y_max = np.min(hist[:, 1]), np.max(hist[:, 1])
        xs, ys = np.linspace(x_min, x_max, num=30), np.linspace(y_min, y_max, num=30)

        xs, ys = np.meshgrid(xs, ys)
        zs = f_true(xs, ys)
        ax.plot_surface(xs, ys, zs, cmap='plasma')

        xs_hist, ys_hist = hist[:, 0], hist[:, 1]
        zs_hist = f_true(xs_hist, ys_hist)
        ax.scatter(xs_hist, ys_hist, zs_hist, c='r', s=10)
        ax.plot(xs_hist, ys_hist, zs_hist, 'r', linewidth=2)
        plt.show()

def plot_errors(histories, labels, x_true):
    for h, l in zip(histories, labels):
        plt.plot(np.abs(h - x_true), label=l)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    f = lambda x: x ** 3 - 10 * x ** 2
    df = lambda x: 3 * x ** 2 - 20 * x
    x_0 = 10.0
    n = 100
    lr = 0.05

    histories = [
        gd(df, x_0, n, lr),
        sgd(df, x_0, n, lr, 1),
        momentum(df, x_0, n, 0.5, lr),
        rms_propagation(df, x_0, n, 0.1, lr, 1e-6),
        adam(df, x_0, n, lr, 0.6, 0.3, 1e-6)]

    labels = ['gd', 'sgd', 'momentum', 'rms_prop', 'adam']

    plot_errors(histories, labels, 0)


if __name__ == '__main__':
    main()