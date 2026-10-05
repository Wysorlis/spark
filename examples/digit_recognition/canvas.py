# coding: utf-8

import tkinter as tk
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from spark.neural_network import NeuralNetwork

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

class DigitRecognitionApp:
    def __init__(self, title: str):
        self.root= tk.Tk()
        
        self.root.title(title)
        self.root.geometry("500x500")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<B1-Motion>", self.draw)

        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1)

        self.slider = tk.Scale(right_frame, from_=1, to=20, orient="horizontal")
        self.slider.set(5)
        self.slider.pack(pady=20)

        self.button = tk.Button(right_frame, text="Reset", command=self.reset_canvas)
        self.button.pack(pady=20)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.grid(row=1, column=0)

        self.label = tk.Label(bottom_frame, text="The predicted digit is :")
        self.label.pack(side="left")

        self.predicted_number = tk.Label(bottom_frame, text="?", font=("Arial", 12, "bold"))
        self.predicted_number.pack(side="left", padx=0)

        self.image = Image.new("L", (280, 280), color=255)
        self.image_draw = ImageDraw.Draw(self.image)

        
    def run(self):
        self.root.mainloop()

    def reset_canvas(self):
        self.canvas.delete("all")

        # Effacer également l'image utilisée pour la prédiction.
        self.image.paste(255, (0, 0, 280, 280))

        self.predicted_number.config(text="?")

    def draw(self, event):
        radius = self.slider.get()

        bounds = (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        )
        
        self.canvas.create_oval(
            *bounds,
            fill="black",
            outline="black",
        )
        
        self.image_draw.ellipse(bounds, fill=0)

        self.predict()

        
    def predict(self):
        # Réduire le dessin aux dimensions attendues par MNIST.
        small_image = self.image.resize(
            (28, 28),
            resample=Image.Resampling.LANCZOS,
        )

        pixels = np.asarray(small_image, dtype=np.float32)

        # Canvas : noir sur blanc.
        # MNIST : clair sur noir, avec des valeurs entre 0 et 1.
        inputs = (255.0 - pixels) / 255.0

        probabilities = self.nn.forward(inputs.flatten())
        digit = int(np.argmax(probabilities))

        self.predicted_number.config(text=str(digit))

    def connect_network(self):
        NN_PATH = Path(__file__).parent / "mnist.npz"
        self.nn = NeuralNetwork.load(NN_PATH)


def main():    
    app = DigitRecognitionApp("Spark - Digit recognition")
    app.connect_network()
    # DATA_PATH = Path(__file__).parent / "data.npz"

    # data = np.load(DATA_PATH)
    # image = data["image"].astype(float).flatten() / 255.0
    # prediction = nn.forward(image)

    # print("prediction : ", prediction)
    app.run()

    

if __name__ == "__main__":
    main()