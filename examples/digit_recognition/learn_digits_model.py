# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display_cities
from keras.datasets import mnist



def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    input_size = 28 * 28
    print(X_train)
    # nn = NeuralNetwork(input_size=input_size)
    # nn.add_layer(layer_width=16, activation_function=relu)
    # nn.add_layer(layer_width=16, activation_function=relu)
    # nn.add_layer(layer_width=10, activation_function=softmax)

    # nn.verbose(True)
    # X, Y = data_points

    # mean = X.mean(axis=0)
    # std = X.std(axis=0)

    # X_scaled = (X - mean) / std

    # error = nn.trains_on((X_scaled, Y), 20000)

    # display_cities(data_points, nn, error)


if __name__ == "__main__":
    main()