# coding: utf-8
import numpy as np
from spark.neural_network import NeuralNetwork
from spark.display import display


def data_test() -> list:
    return (np.array([
        [-1.0, -1.0],
        [-1.0, 1.0],
        [0.8, -0.5],
        [1.0, -1.0],
        [0.7, 0.5],
        [0.5, 0.7],
        [1.0, 1.0],
    ]),
    np.array([-1.0,
     -1.0,
     +1.0,
     +1.0,
     -1.0,
     -1.0,
     +1.0,]))




def main():
    data_points = data_test()
    nn = NeuralNetwork(3)


    # nn.verbose(True)
    error = nn.trains_on(data_points, 500)

    display(data_points, nn.weights, error)








if __name__ == "__main__":
    main()