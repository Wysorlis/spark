# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def display(datapoints: list, weight: list, error: list):
    tests, ys = datapoints
    x, y = tests.T
    error = pd.Series(error)
    print(error)

    x_model = np.linspace(x.min() - 10.0, x.max() + 10.0, 50)
    y_model = -(weight[0] * x_model + weight[2]) / weight[1]

    colors = np.array(["blue" if y >= 0.0 else "red" for y in ys ])
    positive = ys >= 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]},)

    ax1.scatter(x[positive], y[positive], marker="o", color=colors[positive])
    ax1.scatter(x[~positive], y[~positive], marker="x", color=colors[~positive])
    ax1.plot(x_model, y_model, color='black')

    ax2.plot(error.index, error.values, color='darkgray')


    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.set_xlim((1.1 * x.min(), 1.1 * x.max()))
    ax1.set_ylim((1.1 * y.min(), 1.1 * y.max()))

    ax1.grid()
    plt.tight_layout()


    plt.show()