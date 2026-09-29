# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display_cities
from data_test import *

def main():
    data_points = data_test_paris_berlin_madrid_london_latitude_longitude()

    nn = NeuralNetwork(input_size=2)
    nn.add_layer(layer_width=4, activation_function=relu)
    # nn.add_layer(layer_width=64, activation_function=relu)
    # nn.add_layer(layer_width=64, activation_function=relu)
    # nn.add_layer(layer_width=64, activation_function=relu)
    nn.add_layer(layer_width=4, activation_function=softmax)

    nn.verbose(True)
    X, Y = data_points

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_scaled = (X - mean) / std

    error = nn.trains_on((X_scaled, Y), 20000)

    display_cities(data_points, nn, error)


if __name__ == "__main__":
    main()