# coding: utf-8

import tkinter as tk

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
        
    def run(self):
        self.root.mainloop()

    def reset_canvas(self):
        self.canvas.delete("all")

    def draw(self, event):
        radius = self.slider.get()

        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="black",
            outline="black",
        )

        self.predicted_number.config(text="7")


def main():    
    app = DigitRecognitionApp("Spark - Digit recognition")
    app.run()

    

if __name__ == "__main__":
    main()