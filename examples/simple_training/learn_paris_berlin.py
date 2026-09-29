# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display_longitude
from data_test import *

def main():
    data_points = data_test_paris_berlin_longitude()

    nn = NeuralNetwork(input_size=1)
    nn.add_layer(layer_width=2, activation_function=relu)
    nn.add_layer(layer_width=2, activation_function=softmax)

    nn.verbose(True)
    error = nn.trains_on(data_points, 20000)

    display_longitude(data_points, nn, error)


if __name__ == "__main__":
    main()