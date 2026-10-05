# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display_digits
from keras.datasets import mnist
from pathlib import Path


def main():
    DATA_PATH = Path(__file__).parent / "data.npz"
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # data = np.load(DATA_PATH)

    input_size = 28 * 28
    # print(data["label"])
    # y = [1.0  if i == data["label"] else 0.0 for i in range(10)] 
    # print(y)

    nn = NeuralNetwork(input_size=input_size)
    nn.add_layer(layer_width=32, activation_function=relu)
    nn.add_layer(layer_width=32, activation_function=relu)
    nn.add_layer(layer_width=16, activation_function=relu)
    nn.add_layer(layer_width=10, activation_function=softmax)

    X_train = X_train.astype(np.float32) / 255.0
    # Cibles compatibles avec ton gradient Softmax + entropie croisée.
    Y_train = np.eye(10, dtype=np.float32)[y_train]

    error = nn.trains_on((X_train, Y_train), 60_000*3)

    nn.save("mnist.npz")

    print(f"Predicted value is : {nn.forward(X_test[0].flatten())} correct value is : {y_test[0]}")
    # print(y_test[0])

    # display_digits((X_test, y_test), nn, error)


if __name__ == "__main__":
    main()