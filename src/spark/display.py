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
    # ax1.plot(x_model, y_model, color='black')
    print(error)
    # ax2.plot(error.index, error.values, color='black')


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

def display_xor(datapoints, network, error):
    tests, ys = datapoints
    tests = np.asarray(tests)
    ys = np.asarray(ys)
    x, y = tests.T

    # Convertir les cibles en indices de classes : 0 ou 1.
    if ys.ndim == 2 and ys.shape[1] == 2:
        labels = ys.argmax(axis=1)
    else:
        labels = (ys.reshape(-1) > 0).astype(int)

    # Grille couvrant le plan autour des données.
    margin = 0.5
    x_grid = np.linspace(x.min() - margin, x.max() + margin, 400)
    y_grid = np.linspace(y.min() - margin, y.max() + margin, 400)
    xx, yy = np.meshgrid(x_grid, y_grid)

    points = np.column_stack((xx.ravel(), yy.ravel()))

    predictions = np.array([
        network.forward(point) for point in points
    ])

    # Probabilité de la classe 1 à chaque point de la grille.
    probability = predictions[:, 1].reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # Régions : rouge pour la classe 0, bleu pour la classe 1.
    ax1.contourf(
        xx, yy, probability,
        levels=[0, 0.5, 1],
        colors=["red", "blue"],
        alpha=0.2,
    )

    # Frontière : les deux classes sont équiprobables.
    if probability.min() < 0.5 < probability.max():
        ax1.contour(
            xx, yy, probability,
            levels=[0.5],
            colors="black",
            linewidths=2,
        )

    for label, color, marker in [(0, "red", "x"), (1, "blue", "o")]:
        mask = labels == label
        ax1.scatter(
            x[mask], y[mask],
            color=color,
            marker=marker,
            label=f"Classe {label}",
        )

    ax1.set(
        xlabel="x",
        ylabel="y",
        xlim=(x_grid.min(), x_grid.max()),
        ylim=(y_grid.min(), y_grid.max()),
        title="Régions prédites et frontière de décision",
    )
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Une perte scalaire par étape.
    losses = np.asarray(error)
    if losses.ndim == 2 and losses.shape[1] == 1:
        losses = losses[:, 0]
    if losses.ndim != 1:
        raise ValueError("error doit contenir une perte scalaire par étape.")

    ax2.plot(np.arange(len(losses)), losses, color="black")
    ax2.set(xlabel="Étape", ylabel="Perte")
    ax2.grid(alpha=0.3)

    plt.show()


def display_longitude(datapoints, network, error):
    tests, ys = datapoints
    tests = np.asarray(tests)
    ys = np.asarray(ys)
    x = tests.T[0]

    # Convertir les cibles en indices de classes : 0 ou 1.
    if ys.ndim == 2 and ys.shape[1] == 2:
        labels = ys.argmax(axis=1)
    else:
        labels = (ys.reshape(-1) > 0).astype(int)

    # Grille couvrant le plan autour des données.
    margin = 0.5
    x_grid = np.linspace(0 - margin, 15 + margin, 400)
    y_grid = np.linspace(0 - margin, 5 + margin, 400)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # points = np.column_stack((xx.ravel(), yy.ravel()))
    points = xx.ravel()
    # print(xx.ravel())
    # print(points[0])

    predictions = np.array([
        network.forward([point]) for point in points
    ])

    # Probabilité de la classe 1 à chaque point de la grille.
    probability = predictions[:, 1].reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # Régions : rouge pour la classe 0, bleu pour la classe 1.
    ax1.contourf(
        xx, yy, probability,
        levels=[0, 0.5, 1],
        colors=["red", "blue"],
        alpha=0.2,
    )

    # Frontière : les deux classes sont équiprobables.
    if probability.min() < 0.5 < probability.max():
        ax1.contour(
            xx, yy, probability,
            levels=[0.5],
            colors="black",
            linewidths=2,
        )

    for label, color, marker in [(0, "red", "x"), (1, "blue", "o")]:
        mask = labels == label
        label = "Paris" if label == 0 else "Berlin"
        ax1.scatter(
            x[mask], [2.5 for v in x[mask]],
            color=color,
            marker=marker,
            label=f"{label}",
        )

    ax1.set(
        xlabel="x",
        ylabel="y",
        xlim=(x_grid.min(), x_grid.max()),
        ylim=(y_grid.min(), y_grid.max()),
        title="Régions prédites et frontière de décision",
    )
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Une perte scalaire par étape.
    losses = np.asarray(error)
    if losses.ndim == 2 and losses.shape[1] == 1:
        losses = losses[:, 0]
    if losses.ndim != 1:
        raise ValueError("error doit contenir une perte scalaire par étape.")

    ax2.plot(np.arange(len(losses)), losses, color="black")
    ax2.set(xlabel="Étape", ylabel="Perte")
    ax2.grid(alpha=0.3)

    plt.show()


def display_longitude_pbm(datapoints, network, error):
    tests, ys = datapoints
    tests = np.asarray(tests)
    ys = np.asarray(ys)
    x = tests[:, 0]

    # Ordre des sorties dans le jeu de données.
    cities = ["Madrid", "Paris", "Berlin"]
    colors = ["red", "blue", "green"]
    markers = ["x", "o", "^"]

    if ys.ndim == 2 and ys.shape[1] > 1:
        labels = ys.argmax(axis=1)
    else:
        labels = ys.reshape(-1).astype(int)

    if len(labels) != len(x) or np.any((labels < 0) | (labels >= 3)):
        raise ValueError("Les classes attendues sont 0, 1 et 2.")

    margin = max(float(np.ptp(x)) * 0.1, 0.5)
    x_grid = np.linspace(x.min() - margin, x.max() + margin, 400)

    # Une seule prediction par longitude : la hauteur est decorative.
    predictions = np.asarray([network.forward([value]) for value in x_grid])
    if predictions.shape != (len(x_grid), 3):
        raise ValueError("Le réseau doit renvoyer trois sorties par longitude.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Les prédictions contiennent des NaN ou des infinis.")

    predicted_labels = predictions.argmax(axis=1)
    xx, yy = np.meshgrid(x_grid, [0.0, 1.0])
    regions = np.tile(predicted_labels, (2, 1))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # La couleur represente la classe gagnante, pas un seuil de probabilite.
    ax1.contourf(
        xx, yy, regions,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=colors,
        alpha=0.2,
        antialiased=False,
    )

    # Frontieres approximatives, uniquement lorsque la classe gagnante change.
    changes = np.flatnonzero(predicted_labels[:-1] != predicted_labels[1:])
    for i in changes:
        left_class = predicted_labels[i]
        right_class = predicted_labels[i + 1]
        delta_left = predictions[i, left_class] - predictions[i, right_class]
        delta_right = predictions[i + 1, left_class] - predictions[i + 1, right_class]
        boundary = x_grid[i] + (x_grid[i + 1] - x_grid[i]) * (
            delta_left / (delta_left - delta_right)
        )
        ax1.axvline(boundary, color="black", linestyle="--", linewidth=1)

    for label, (city, color, marker) in enumerate(zip(cities, colors, markers)):
        mask = labels == label
        ax1.scatter(
            x[mask], np.full(mask.sum(), 0.5),
            color=color, marker=marker, label=city, zorder=3,
        )

    ax1.set(
        xlabel="Longitude",
        xlim=(x_grid.min(), x_grid.max()),
        ylim=(0, 1),
        title="Madrid, Paris, Berlin — régions prédites",
    )
    ax1.set_yticks([])  # Le modele ne recoit pas de latitude.
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    losses = np.asarray(error)
    if losses.ndim == 2 and losses.shape[1] == 1:
        losses = losses[:, 0]
    if losses.ndim != 1:
        raise ValueError("error doit contenir une perte scalaire par étape.")

    ax2.plot(np.arange(len(losses)), losses, color="black")
    ax2.set(xlabel="Étape", ylabel="Perte")
    ax2.grid(alpha=0.3)
    plt.show()
