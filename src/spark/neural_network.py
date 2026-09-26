# coding: utf-8

import numpy as np
import pandas as pd

class Layer:
    def __init__(self, width: int, previous_layer_width: int):
        self.width = width
        self.weights = np.zeros((width, previous_layer_width))
        self.biases = np.zeros(width)
        print(self.weights.shape)
        # exit()


class NeuralNetwork:
    def __init__(self, input_size: int):
        self.inputs_size = input_size
        self.weights = np.zeros(input_size) 
        self.layers = []
        self.learning_rate = 0.1
        self.is_verbose = False

        self.data = {
            'x': [],
            'y': [],
            'w': [],
            'b': [],
            'h': [],
            'y_hat': [],
            'y': [],
            'grad_weights': [],
            'grad_biases': [],
            'loss': [],
        }

    def add_layer(self, layer_width: Layer):
        if not self.layers:
            self.layers.append(Layer(width=layer_width, previous_layer_width=self.inputs_size))
            return
        
        previous_layer_width = self.layers[-1].width
        self.layers.append(
            Layer(width=layer_width,
                  previous_layer_width=previous_layer_width
                 ))

    def forward(self, input_value):
        if input_value != self.inputs_size:
            raise ValueError("L'entrée doit avoir la même taile que la même forme que ({self.inputs_size})")

        out = self.weights

        for layer in layers:
            layer_results = layer.weights


    def verbose(self, is_verbose: bool):
        self.is_verbose = is_verbose

    def trains_on(self, datapoints, n_steps=50):
        tests, ys = datapoints
        sample_size = len(tests)

        pd.set_option('display.width', None)
        pd.set_option("display.float_format", lambda x: f"{x:.3g}")

        for layer in self.layers:
            # weights = np.array([weight for layer in self.layers for weight in layer.weights])

            # print(weights)
            
            for i in range(n_steps):
                test = tests[i % sample_size]
                y = ys[i % sample_size]

                x = test
                # print(layer.weights)

                h = np.dot(layer.weights, x) + layer.biases

                # Soustraire c = max(h) évite le débordement des exponentielles.
                # Cela ne change pas Softmax : le facteur exp(-c) s'annule.
                #
                # exp(h_i - c) / sum_j(exp(h_j - c))
                # = [exp(-c) * exp(h_i)] / [exp(-c) * sum_j(exp(h_j))]
                # = exp(h_i) / sum_j(exp(h_j))
                exp_h = np.exp(h - np.max(h))
                y_hat = exp_h / exp_h.sum()

                loss = - np.log(y_hat)

                dLdh = y_hat - y

                grad_weights = np.outer(dLdh, x)
                grad_biases = dLdh

                # print("layer.weights.shape", layer.weights.shape)
                # print("layer.weights", layer.weights)

                # print("grad_weights.shape", grad_weights.shape)
                # print("grad_biagrad_weightsses", grad_weights)


                self.data['x'].append(np.round(x, 3))
                self.data['y'].append(y)
                self.data['w'].append(np.round(layer.weights, 3))
                self.data['b'].append(np.round(layer.biases, 3))
                self.data['h'].append(np.round(h, 3))
                self.data['y_hat'].append(np.round(y_hat, 3))
                self.data['grad_weights'].append(np.round(grad_weights, 3))
                self.data['grad_biases'].append(np.round(grad_biases, 3))
                self.data['loss'].append(np.round(loss, 3))

                layer.weights -= self.learning_rate * grad_weights
                layer.biases -= self.learning_rate * grad_biases

                if self.is_verbose:
                    self.print()

            # self.weights = weights

        return self.data['loss']
    
    def print(self):
        print(self.data)
        print(pd.DataFrame(self.data))



