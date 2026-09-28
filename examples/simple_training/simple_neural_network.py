# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork, Layer, softmax, relu
from spark.display import display, new_display, new_display_network
from data_test import *




def main():
    data_points = data_test_new()
    # data_points = data_test_more()

    nn = NeuralNetwork(input_size=1)
    nn.add_layer(layer_width=2)
    nn.add_layer(layer_width=2, activation_function=softmax)
    # nn.add_layer(layer_width=3)
    layer = nn.layers[0]
    layer.weights[0] = +2
    layer.weights[1] = +1


    # nn.verbose(True)
    error = nn.trains_on(data_points, 10_000)

    point = [8]
    probabilities = nn.forward(point)

    index = np.argmax(probabilities)
    guess = ["Paris", "Berlin"][index]
    confidence = probabilities[index]

    print(f"The network guessed that the point {point} is close to {guess} ({np.round(probabilities, 4)})")
    # nn.print()

    new_display_network(data_points, nn, nn.data)


if __name__ == "__main__":
    main()