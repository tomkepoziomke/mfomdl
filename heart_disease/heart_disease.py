import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_coefficients(coeffs, feature_names):
    width = 0.9 / len(feature_names)
    multiplier = 0

    x = np.arange(len(feature_names))

    fig, ax = plt.subplots()
    ax.grid()
    ax.set_yscale("symlog")

    for feature, attrs in coeffs.items():
        ax.bar(x + width * multiplier, attrs, width, label=feature, zorder=10)
        ax.legend()
        multiplier += 1
    ax.set_xticks(x + width, feature_names)
    plt.show()

def standard_models(X, y, feature_names):
    types = [LinearRegression, Ridge, Lasso, ElasticNet]
    labels = ["LR", "Ridge", "Lasso", "ElasticNet"]
    models = []

    for model, label in zip(types, labels):
        m = model()
        m.fit(X, y)
        models.append(m)

        err = mean_squared_error(m.predict(X), y)
        print(f"{label} MSE: {err}")

    coeffs = {}
    for i, label in enumerate(labels):
        coeffs[label] = models[i].coef_

    plot_coefficients(coeffs, feature_names)

def standard_scaled(X, y, feature_names):
    scaler = StandardScaler()
    scaler.fit(X, y)
    X_scale = scaler.transform(X)
    standard_models(X_scale, y, feature_names)

def ridge(X, y, feature_names):
    penalties = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    labels = [str(p) for p in penalties]
    models = []

    for penalty, label in zip(penalties, labels):
        m = Ridge(alpha=penalty)
        m.fit(X, y)
        models.append(m)

        err = mean_squared_error(m.predict(X), y)
        print(f"Ridge alpha={label} MSE: {err}")

    coeffs = {}
    for i, label in enumerate(labels):
        coeffs[label] = models[i].coef_

    plot_coefficients(coeffs, feature_names)
def main():
    data = load_diabetes()
    X, y, feature_names = data.data, data.target, data.feature_names

    standard_models(X, y, feature_names)
    standard_scaled(X, y, feature_names)
    ridge(X, y, feature_names)

if __name__ == '__main__':
    main()




