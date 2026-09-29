# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display_xor
from data_test import *

def main():
    data_points = data_test_xor()

    nn = NeuralNetwork(input_size=2)
    nn.add_layer(layer_width=8, activation_function=relu)
    nn.add_layer(layer_width=2, activation_function=softmax)

    nn.verbose(True)
    # error = nn.trains_on(data_points, 20_000)
    error = nn.trains_on(data_points, 2_000)

    display_xor(data_points, nn, error)


if __name__ == "__main__":
    main()