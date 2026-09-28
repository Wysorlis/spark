# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def display(datapoints: list, weight: list, error: list):
    tests, ys = datapoints
    x, y = tests.T
    error = pd.Series(error)

    x_model = np.linspace(x.min() - 10.0, x.max() + 10.0, 50)
    y_model = -(weight[0] * x_model + weight[2]) / weight[1]

    colors = np.array(["blue" if y >= 0.0 else "red" for y in ys ])
    positive = ys >= 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]},)

    ax1.scatter(x[positive], y[positive], marker="o", color=colors[positive])
    ax1.scatter(x[~positive], y[~positive], marker="x", color=colors[~positive])
    ax1.plot(x_model, y_model, color='black')

    ax2.plot(error.index, error.values, color='black')


    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.set_xlim((1.1 * x.min(), 1.1 * x.max()))
    ax1.set_ylim((1.1 * y.min(), 1.1 * y.max()))

    ax1.grid()
    plt.tight_layout()


    plt.show()


def new_display(datapoints, weights, biases, history):
    tests, targets = datapoints
    x = np.asarray(tests)[:, 0]
    targets = np.asarray(targets)

    # Prédictions du modèle final sur une grille de longitudes
    margin = max(float(np.ptp(x)) * 0.1, 0.5)
    x_model = np.linspace(x.min() - margin, x.max() + margin, 400)

    # (400, 1) @ (1, 2) → (400, 2)
    scores = x_model[:, None] @ weights.T + biases

    # Softmax pour chaque point de la grille
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    for i, (name, color) in enumerate([
        ("Paris", "red"),
        ("Berlin", "blue"),
    ]):
        ax1.plot(
            x_model, probabilities[:, i],
            color=color, label=f"P({name})",
        )
        mask = targets[:, i] == 1

        ax1.scatter(
            x[mask],
            targets[mask, i],
            color=color,
            marker="x",
        )

    ax1.axhline(0.5, color="gray", linestyle=":", label="Seuil 0.5")

    # Frontière : score Paris = score Berlin
    delta_w = weights[0, 0] - weights[1, 0]
    delta_b = biases[0] - biases[1]

    if not np.isclose(delta_w, 0):
        boundary = -delta_b / delta_w
        if x_model.min() <= boundary <= x_model.max():
            ax1.axvline(
                boundary, color="black", linestyle="--",
                label=f"Frontière : {boundary:.3f}",
            )

    ax1.set(
        xlabel="Longitude",
        ylabel="Probabilité",
        ylim=(-0.05, 1.05),
        title="Prédictions après entraînement — croix : cibles",
    )
    ax1.legend()

    # Ton historique stocke -log(p) pour chacune des deux classes.
    # On sélectionne la classe cible de chaque étape.
    losses_by_class = np.asarray(history["loss"])
    labels = np.asarray(history["y"]).argmax(axis=1)
    losses = losses_by_class[np.arange(len(labels)), labels]

    ax2.plot(np.arange(1, len(losses) + 1), losses, color="black")
    ax2.set(
        xlabel="Étape d'entraînement",
        ylabel="Perte",
        title="Perte sur l'exemple traité à chaque étape",
    )

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)

    plt.show()

def new_display_network(datapoints, network, history):
    tests, targets = datapoints
    x = np.asarray(tests)[:, 0]
    targets = np.asarray(targets)

    # Prédictions du modèle final sur une grille de longitudes
    margin = max(float(np.ptp(x)) * 0.1, 0.5)
    x_model = np.linspace(x.min() - margin, x.max() + margin, 400)

    # Chaque longitude traverse toutes les couches et leurs activations.
    predictions = []
    for longitude in x_model:
        current = np.array([longitude])
        for layer in network.layers:
            z = layer.weights @ current + layer.biases
            current = layer.activation_function(z)
        predictions.append(current.copy())

    probabilities = np.asarray(predictions)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    for i, (name, color) in enumerate([
        ("Paris", "red"),
        ("Berlin", "blue"),
    ]):
        ax1.plot(
            x_model, probabilities[:, i],
            color=color, label=f"P({name})",
        )
        mask = targets[:, i] == 1

        ax1.scatter(
            x[mask],
            targets[mask, i],
            color=color,
            marker="x",
        )

    ax1.axhline(0.5, color="gray", linestyle=":", label="Seuil 0.5")

    # Frontières : changements de classe entre deux points de la grille.
    # Une égalité partout ne définit pas une frontière unique.
    difference = probabilities[:, 0] - probabilities[:, 1]
    nonzero = np.flatnonzero(difference != 0)
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        if np.sign(difference[left]) == np.sign(difference[right]):
            continue
        if right == left + 1:
            boundary = x_model[left] - difference[left] * (
                x_model[right] - x_model[left]
            ) / (difference[right] - difference[left])
        else:
            # Milieu d'une éventuelle zone d'égalité entre les classes.
            boundary = (x_model[left + 1] + x_model[right - 1]) / 2
        ax1.axvline(
            boundary, color="black", linestyle="--",
            label=f"Frontière approximative : {boundary:.3f}",
        )

    ax1.set(
        xlabel="Longitude",
        ylabel="Probabilité",
        ylim=(-0.05, 1.05),
        title="Prédictions après entraînement — croix : cibles",
    )
    ax1.legend()

    # Ton historique stocke -log(p) pour chacune des deux classes.
    # On sélectionne la classe cible de chaque étape.
    losses = np.asarray(history["loss"])
    if losses.ndim == 2:
        labels = np.asarray(history["y"]).argmax(axis=1)
        losses = losses[np.arange(len(labels)), labels]

    ax2.plot(np.arange(1, len(losses) + 1), losses, color="black")
    ax2.set(
        xlabel="Étape d'entraînement",
        ylabel="Perte",
        title="Perte sur l'exemple traité à chaque étape",
    )

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)

    plt.show()