# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


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


def display_cities(datapoints, network, error):
    tests, ys = datapoints
    tests = np.asarray(tests)
    ys = np.asarray(ys)

    cities = ["Madrid", "Paris", "Berlin", "London"]
    colors = ["red", "blue", "green", "orange"]
    markers = ["x", "o", "^", "s"]
    n_classes = len(cities)

    if tests.ndim != 2 or tests.shape[1] != 2:
        raise ValueError("Les entrées doivent être [latitude, longitude].")

    latitude = tests[:, 0]
    longitude = tests[:, 1]

    if ys.ndim == 2 and ys.shape[1] > 1:
        if ys.shape[1] != n_classes:
            raise ValueError("Les cibles doivent avoir quatre composantes.")
        labels = ys.argmax(axis=1)
    else:
        labels = ys.reshape(-1).astype(int)

    if len(labels) != len(tests) or np.any(
        (labels < 0) | (labels >= n_classes)
    ):
        raise ValueError("Les classes attendues sont 0, 1, 2 et 3.")

    # Grille géographique autour des données.
    margin_lon = max(float(np.ptp(longitude)) * 0.1, 0.5)
    margin_lat = max(float(np.ptp(latitude)) * 0.1, 0.5)

    x_grid = np.linspace(
        longitude.min() - margin_lon,
        longitude.max() + margin_lon,
        250,
    )
    y_grid = np.linspace(
        latitude.min() - margin_lat,
        latitude.max() + margin_lat,
        250,
    )
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Le réseau attend [latitude, longitude], donc [yy, xx].
    points = np.column_stack((yy.ravel(), xx.ravel()))
    X, Y = datapoints

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    predictions = np.asarray([
        network.forward((point - mean) / std)
        for point in points
    ])

    if predictions.shape != (len(points), n_classes):
        raise ValueError("Le réseau doit renvoyer quatre sorties par point.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Les prédictions contiennent des NaN ou des infinis.")

    regions = predictions.argmax(axis=1).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # Régions correspondant à la classe gagnante.
    ax1.contourf(
        xx, yy, regions,
        levels=np.arange(n_classes + 1) - 0.5,
        colors=colors,
        alpha=0.2,
        antialiased=False,
    )

    # Contours des régions : approximation à la résolution de la grille.
    for label in range(n_classes):
        region_mask = (regions == label).astype(float)
        if region_mask.min() != region_mask.max():
            ax1.contour(
                xx, yy, region_mask,
                levels=[0.5],
                colors="black",
                linewidths=0.8,
            )

    for label, (city, color, marker) in enumerate(
        zip(cities, colors, markers)
    ):
        mask = labels == label
        ax1.scatter(
            longitude[mask],
            latitude[mask],
            color=color,
            marker=marker,
            label=city,
            zorder=3,
        )

    ax1.set(
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=(x_grid.min(), x_grid.max()),
        ylim=(y_grid.min(), y_grid.max()),
        title="Madrid, Paris, Berlin, London — régions prédites",
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    losses = np.asarray(error)
    if losses.ndim == 2 and losses.shape[1] == 1:
        losses = losses[:, 0]
    if losses.ndim != 1:
        raise ValueError("error doit contenir une perte scalaire par étape.")

    ax2.plot(np.arange(len(losses)), losses, color="black")
    ax2.set(xlabel="Étape", ylabel="Perte")
    ax2.grid(alpha=0.3)

    plt.show()



def show_digit(digit, label):
    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 8),
        constrained_layout=True,
    )

    digit.reshape(28, -1)
    ax.imshow(digit, cmap="gray")
    ax.set_title(f"Label : {label}")

    plt.show()

def display_digits(datapoints, network, error, n_digits=10):
    tests, ys = datapoints

    tests = np.asarray(tests)
    ys = np.asarray(ys)

    if tests.ndim not in (2, 3):
        raise ValueError(
            "Les entrées doivent être de forme "
            "(n, 28, 28) ou (n, 784)."
        )

    if tests.ndim == 2 and tests.shape[1] != 28 * 28:
        raise ValueError("Chaque entrée doit contenir 784 pixels.")

    if tests.ndim == 3 and tests.shape[1:] != (28, 28):
        raise ValueError("Les images doivent être de taille 28 × 28.")

    # Labels : accepte aussi bien
    # [5, 0, 4, ...]
    # que du one-hot :
    # [[0, 0, ..., 1, ...], ...]
    if ys.ndim == 2:
        labels = ys.argmax(axis=1)
    else:
        labels = ys.astype(int)

    if len(labels) != len(tests):
        raise ValueError(
            "Il doit y avoir autant de labels que d'images."
        )

    n_digits = min(n_digits, len(tests))

    predictions = []

    for digit in tests[:n_digits]:
        X = digit.flatten().astype(float)

        # Même normalisation que pendant l'entraînement.
        if X.max() > 1.0:
            X /= 255.0

        prediction = network.forward(X)
        predictions.append(prediction)

    predictions = np.asarray(predictions)

    if predictions.shape != (n_digits, 10):
        raise ValueError(
            "Le réseau doit renvoyer 10 sorties par image."
        )

    predicted_labels = predictions.argmax(axis=1)

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 7),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        2,
        n_digits,
        height_ratios=[2, 1],
    )

    # ---------------------------------------------------------
    # Chiffres
    # ---------------------------------------------------------

    for i in range(n_digits):
        ax = fig.add_subplot(gs[0, i])

        image = tests[i].reshape(28, 28)

        true_label = labels[i]
        predicted_label = predicted_labels[i]
        confidence = predictions[i, predicted_label]

        ax.imshow(
            image,
            cmap="gray",
        )

        ax.set_title(
            f"{true_label} → {predicted_label}\n"
            f"{confidence:.1%}"
        )

        ax.axis("off")

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    ax_loss = fig.add_subplot(gs[1, :])

    losses = np.asarray(error)

    if losses.ndim == 2 and losses.shape[1] == 1:
        losses = losses[:, 0]

    if losses.ndim != 1:
        raise ValueError(
            "error doit contenir une perte scalaire par étape."
        )

    ax_loss.plot(
        np.arange(len(losses)),
        losses,
    )

    ax_loss.set(
        xlabel="Étape",
        ylabel="Perte",
        title="Évolution de la perte",
    )

    ax_loss.grid(alpha=0.3)

    plt.show()

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "data.npz"

    # from keras.datasets import mnist

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # np.savez(DATA_PATH, image=X_train[0], label=y_train[0])

    data = np.load(DATA_PATH)
    show_digit(data["image"], data["label"])

