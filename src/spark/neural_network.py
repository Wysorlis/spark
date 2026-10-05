# coding: utf-8

import numpy as np
import pandas as pd


def softmax(layer_ouput):
    # Soustraire c = max(h) évite le débordement des exponentielles.
    # Cela ne change pas Softmax : le facteur exp(-c) s'annule.
    #
    # exp(h_i - c) / sum_j(exp(h_j - c))
    # = [exp(-c) * exp(h_i)] / [exp(-c) * sum_j(exp(h_j))]
    # = exp(h_i) / sum_j(exp(h_j))
    exp_layer_ouput = np.exp(layer_ouput - np.max(layer_ouput))
    y_hat = exp_layer_ouput / exp_layer_ouput.sum()

    return y_hat

def identity(z):
    return z

def relu(layer_ouput):
    return np.maximum(0, layer_ouput)

# def relu_derivative(layer_ouput):
#     return np.maximum(layer_ouput)
import json

ACTIVATIONS = {
    "identity": identity,
    "relu": relu,
    "softmax": softmax,
}


class Layer:
    def __init__(self, width: int, previous_layer_width: int, activation_function: callable):
        self.width = width
        self.weights = np.zeros((width, previous_layer_width))
        self.biases = np.zeros(width)
        self.activation_function = activation_function
        self._initialise_weight_values(previous_layer_width)

    def _initialise_weight_values(self, previous_layer_width):
        rng = np.random.default_rng()

        self.weights = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / previous_layer_width),
            size=(self.width, previous_layer_width),
        )
        self.biases = np.zeros(self.width)


class NeuralNetwork:
    def __init__(self, input_size: int):
        self.inputs_size = input_size
        self.weights = np.zeros(input_size) 
        self.layers = []
        self.learning_rate = 0.01
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

    def add_layer(self, layer_width: int, activation_function: callable=identity):
        # Si on a pas déjà des Layers alors la taille correspond aux paramètres d'entrées x
        if not self.layers:
            self.layers.append(Layer(width=layer_width, previous_layer_width=self.inputs_size, activation_function=activation_function))
            return
        
        previous_layer_width = self.layers[-1].width
        self.layers.append(
            Layer(width=layer_width,
                  previous_layer_width=previous_layer_width,
                  activation_function=activation_function
                 ))

    def forward(self, input_values):
        current = np.asarray(input_values, dtype=float)

        if current.shape != (self.inputs_size,):
            raise ValueError(
                f"Forme attendue : ({self.inputs_size},), "
                f"forme reçue : {current.shape}."
            )
        
        for layer in self.layers:
            z = layer.weights @  current + layer.biases
            current = layer.activation_function(z)
            
        return current

    def verbose(self, is_verbose: bool):
        self.is_verbose = is_verbose

    def trains_on(self, datapoints, n_steps=50):
        tests, ys = datapoints
        sample_size = len(tests)
        # print("sample_size", sample_size)

        pd.set_option('display.width', None)
        pd.set_option("display.float_format", lambda x: f"{x:.3g}")
        
        for i in range(n_steps):
            sample_index = i % sample_size

            X = tests[sample_index].flatten()
            y = ys[sample_index]

            # 1) Calcul de la prédiction
            current_layer = X

            for layer in self.layers:
                layer.input = current_layer

                # print("current_layer")
                # print(current_layer.shape)
                # print(current_layer)

                # print("layer.weights")
                # print(layer.weights.shape)
                # print(layer.weights.shape)
                # print(layer.weights)
                layer.z = layer.weights @  current_layer + layer.biases

                layer.output = layer.activation_function(layer.z)
                current_layer = layer.output

            y_hat = current_layer

            # loss = - np.log(y_hat)
            z = self.layers[-1].z
            shifted = z - np.max(z)
            log_probs = shifted - np.log(np.exp(shifted).sum())

            loss = -np.sum(y * log_probs)

            # 2) Calculer les gradients
            dLdz = y_hat - y
            for index, layer in reversed(list(enumerate(self.layers))):
                
                layer.grad_weights = np.outer(dLdz, layer.input)
                layer.grad_biases = dLdz.copy()

                if index == 0:
                    break  # Plus de couche précédente à traiter.

                # L'entrée de cette couche est la sortie de la précédente.
                dLda = layer.weights.T @ dLdz

                previous = self.layers[index - 1]
                activation = previous.activation_function
                a = previous.output

                if activation is identity:
                    dLdz = dLda

                elif activation is relu:
                    dLdz = dLda * (previous.z > 0)

                elif activation is softmax:
                    # Softmax cachée : ses sorties sont interdépendantes.
                    dLdz = a * (dLda - np.dot(dLda, a))

            # 3) Mises à jours des poids et biais
            for layer in self.layers:
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.biases -= self.learning_rate * layer.grad_biases


            self.data['x'].append(np.round(X, 3))
            self.data['y'].append(y)
            # self.data['w'].append(np.round(layer.weights, 3))
            # self.data['b'].append(np.round(layer.biases, 3))
            # self.data['h'].append(np.round(h, 3))
            self.data['y_hat'].append(np.round(y_hat, 3))
            # self.data['grad_weights'].append(np.round(grad_weights, 3))
            # self.data['grad_biases'].append(np.round(grad_biases, 3))
            self.data["loss"].append(float(loss))


            if self.is_verbose:
                # print(np.max(loss))
                print(f"Étape {i + 1} : perte = {np.max(loss):.6f}")

        return pd.Series(self.data['loss'])
    
    def print(self):
        print(self.data)
        print(pd.DataFrame(self.data))

    def save(self, path):
        config = {
            "format_version": 1,
            "input_size": self.inputs_size,
            "learning_rate": self.learning_rate,
            "layers": [],
        }
        arrays = {}

        for i, layer in enumerate(self.layers):
            # Retrouver le nom associé à la fonction.
            activation_name = next(
                (
                    name
                    for name, function in ACTIVATIONS.items()
                    if function is layer.activation_function
                ),
                None,
            )

            if activation_name is None:
                raise ValueError(
                    f"Activation non enregistrée pour la couche {i}."
                )

            config["layers"].append({
                "width": layer.width,
                "activation": activation_name,
            })

            arrays[f"weights_{i}"] = layer.weights
            arrays[f"biases_{i}"] = layer.biases

        # Architecture en JSON, paramètres en tableaux NumPy.
        arrays["config"] = np.array(json.dumps(config))

        # Ouvrir le fichier évite que NumPy ajoute automatiquement ".npz".
        with open(path, "wb") as file:
            np.savez_compressed(file, **arrays)


    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(data["config"].item())

            if config["format_version"] != 1:
                raise ValueError("Version de sauvegarde non prise en charge.")

            network = cls(input_size=config["input_size"])
            network.learning_rate = config["learning_rate"]

            for i, layer_config in enumerate(config["layers"]):
                activation_name = layer_config["activation"]

                if activation_name not in ACTIVATIONS:
                    raise ValueError(
                        f"Activation inconnue : {activation_name}."
                    )

                network.add_layer(
                    layer_width=layer_config["width"],
                    activation_function=ACTIVATIONS[activation_name],
                )

                layer = network.layers[-1]
                weights = data[f"weights_{i}"]
                biases = data[f"biases_{i}"]

                if (
                    weights.shape != layer.weights.shape
                    or biases.shape != layer.biases.shape
                ):
                    raise ValueError(
                        f"Paramètres incompatibles avec la couche {i}."
                    )

                # Remplacer l'initialisation aléatoire par les paramètres appris.
                layer.weights = weights.copy()
                layer.biases = biases.copy()

        return network



