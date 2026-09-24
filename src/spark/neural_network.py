# coding: utf-8

import numpy as np
import pandas as pd


class NeuralNetwork():
    def __init__(self, N):
        self.weights = np.zeros(N) 
        self.rate = 0.01
        self.verbose = False

        self.data = {
            'x': [],
            'y': [],
            'w': [],
            'y_hat': [],
            'dy': [],
            'E': [],
            'dEdw': [],
            'gradient': [],
        }

    def trains_on(self, verbose: bool):
        self.verbose = verbose

    def trains_on(self, datapoints, n_steps=50):
        tests, ys = datapoints
        sample_size = len(tests)

        pd.set_option('display.width', None)
        pd.set_option("display.float_format", lambda x: f"{x:.3g}")

        for i in range(n_steps):
            test = tests[i % sample_size]
            y = ys[i % sample_size]

            x = np.append(test, 1)

            y_hat = np.dot(self.weights, x)

            dy = y - y_hat
            
            E = dy**2.0

            dEdw = -2.0 * dy * x

            gradient = 0.5 * dEdw

            self.weights -= self.rate * gradient

            self.data['x'].append(np.round(x, 3))
            self.data['y'].append(y)
            self.data['w'].append(np.round(self.weights, 3))
            self.data['y_hat'].append(y_hat)
            self.data['dy'].append(dy)
            self.data['E'].append(E)
            self.data['dEdw'].append(np.round(dEdw, 3))
            self.data['gradient'].append(np.round(gradient, 3))

            if self.verbose:
                print(pd.DataFrame(self.data))

        return self.data['y_hat']
    
    def print(self):
        print(pd.DataFrame(data))



