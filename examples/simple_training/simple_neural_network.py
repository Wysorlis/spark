# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer
from spark.display import display, new_display
from data_test import *




def main():
    data_points = data_test_new()
    # data_points = data_test_more()

    nn = NeuralNetwork(input_size=1)
    nn.add_layer(layer_width=2)
    # nn.add_layer(layer_width=3)
    layer = nn.layers[0]
    layer.weights[0] = -1
    layer.weights[1] = +1


    # nn.verbose(True)
    error = nn.trains_on(data_points, 100_000)
    
    nn.print()

    new_display(data_points,
                layer.weights,
                layer.biases,
                nn.data,)


if __name__ == "__main__":
    main()