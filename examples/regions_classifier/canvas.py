# coding: utf-8

import tkinter as tk
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton

from spark.neural_network import NeuralNetwork, Layer, softmax, relu

# programme lancé
#      ↓
# création de la fenêtre
#      ↓
# création des boutons / canvas / labels
#      ↓
# mainloop()
#      ↓
# Tkinter attend :
#     souris
#     clavier
#     clic bouton
#     redimensionnement
#     etc.

class RegionsClassifierApp:
    def __init__(self, title: str):
        self.root= tk.Tk()
        
        self.root.title(title)
        self.root.geometry("1000x800")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        self.points = []
        self.labels = []

        self.canvas = self._setup_canvas()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw()

        self.reset = tk.Button(self.root, text="Reset Network", command=self.reset_network)
        self.reset.pack(pady=20)

        self.training_index = 0

        self.reset_network()
        self.regions = None
        self.regions_need_update = False

        self.root.after(1, self.train_tick)
        self.root.after(2, self.display_tick)

        
    def run(self):
        self.root.mainloop()

    def display_tick(self):
        if self.regions_need_update :
            self.draw_regions()
            self.draw_loss()
            self.regions_need_update  = False

        self.root.after(5, self.display_tick)

    def train_tick(self):
        if self.points:
            # Petit nombre d'étapes pour ne pas bloquer l'interface.
            for _ in range(10):
                index = self.training_index % len(self.points)

                x = np.asarray(self.points[index], dtype=float)
                y = np.asarray(self.labels[index], dtype=float)

                self.nn.train_step(x, y)
                self.training_index += 1

            self.regions_need_update = True

        self.root.after(2, self.train_tick)

    def _setup_canvas(self):
        self.figure = Figure(figsize=(6, 5), layout="none")
        # Marges fixes pour que les graduations ne déplacent pas les graphiques.
        self.figure.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.95, hspace=0.30)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[6, 1])
        self.ax = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1])

        self.loss_line, = self.ax2.plot([], [], color="black")
        self.ax2.set(xlabel="Étape", ylabel="Perte", xlim=(0, 1), ylim=(0, 1))
        self.ax2.grid(alpha=0.3)

        self.ax.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            xlabel="x",
            ylabel="y",
            title="Clic gauche : rouge — clic droit : bleu",
        )
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.3)

        # Deux groupes de points, initialement vides.
        self.scatter_red = self.ax.scatter([], [], color="red", marker="o", label="Classe A", zorder=3)
        self.scatter_blue = self.ax.scatter([], [], color="blue", marker="o", label="Classe B", zorder=3)
        
        self.ax.legend()

        return FigureCanvasTkAgg(self.figure, master=self.root)

    def on_click(self, event):
        # Ignorer les clics hors de la zone des axes.
        if event.inaxes is not self.ax:
            return

        if event.button == MouseButton.LEFT:
            label = [1, 0]
        elif event.button == MouseButton.RIGHT:
            label = [0, 1]
        else:
            return

        # Coordonnées du graphique, et non coordonnées en pixels !
        point = [event.xdata, event.ydata]

        self.points.append(point)
        self.labels.append(label)

        self.refresh_points()

    def refresh_points(self):
        labels = np.asarray(self.labels)

        red_mask = np.all(labels == [1, 0], axis=1)
        blue_mask = np.all(labels == [0, 1], axis=1)

        points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        labels = np.asarray(self.labels, dtype=int)

        self.scatter_red.set_offsets(points[red_mask])
        self.scatter_blue.set_offsets(points[blue_mask])

        # Demander le rafraîchissement de la figure.
        self.canvas.draw_idle()

    def reset_canvas(self):
        self.canvas.delete("all")

        # Effacer également l'image utilisée pour la prédiction.
        self.image.paste(255, (0, 0, 500, 500))

        self.predicted_number.config(text="?")

    def draw_loss(self):
        losses = np.asarray(self.nn.data["loss"])
        self.loss_line.set_data(np.arange(len(losses)), losses)

        if len(losses) > 0:
            self.ax2.relim()
            self.ax2.autoscale(enable=True, axis="both")
            self.ax2.autoscale_view()

        self.canvas.draw_idle()

    def draw_regions(self, colors=["red", "blue"]):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        x_grid = np.linspace(x_min - 0.1, x_max + 0.1, 60)
        y_grid = np.linspace(y_min - 0.1, y_max + 0.1, 60)

        xx, yy = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack((xx.ravel(), yy.ravel()))

        # predictions = np.asarray([self.nn.forward(point) for point in grid_points])
        predictions = self.nn.forward_batch(grid_points)

        probability_blue = predictions[:, 1].reshape(xx.shape)
        # regions = predictions.argmax(axis=1).reshape(xx.shape)

        if self.regions is not None:
            self.regions.remove()

        self.regions = self.ax.contourf(
            xx, yy, probability_blue,
            levels=[0.0, 0.5, 1.0],
            colors=colors,
            alpha=0.2,
            antialiased=False,
        )

        self.canvas.draw_idle()
        
    def reset_network(self):
        self.nn = NeuralNetwork(input_size=2)
        self.nn.add_layer(layer_width=16, activation_function=relu)
        self.nn.add_layer(layer_width=16, activation_function=relu)
        self.nn.add_layer(layer_width=16, activation_function=relu)
        self.nn.add_layer(layer_width=2, activation_function=softmax)

        self.training_index = 0
        self.loss_line.set_data([], [])
        self.ax2.set(xlim=(0, 1), ylim=(0, 1))
        self.regions_need_update = True
        self.canvas.draw_idle()

        
def main():    
    app = RegionsClassifierApp("Spark - Regions classifier")
    app.run()

    

if __name__ == "__main__":
    main()