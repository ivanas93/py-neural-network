import numpy as np

from NeuralLayer import NeuralLayer as NeuralLayer


class NeuralNetwork:
    def __init__(self, topology, activate_function, derivative_activate_function, least_squares,
                 least_squares_derivative, learning_rate):
        self.topology = topology
        self.activate_function = activate_function
        self.derivative_activate_function = derivative_activate_function
        self.least_squares = least_squares
        self.least_squares_derivative = least_squares_derivative
        self.learning_rate = learning_rate
        self.network = self.create_network()

    def create_network(self):
        network = []
        for la, layer in enumerate(self.topology[:-1]):
            network.append(NeuralLayer(self.topology[la], self.topology[la + 1], self.activate_function,
                                       self.derivative_activate_function))
        return network

    def network(self):
        return self.network

    def train(self, x, y, train=True):
        out = [(None, x)]

        # Forward pass
        for la, layer in enumerate(self.network):
            # Weighted sum
            z = out[-1][1] @ self.network[la].W + self.network[la].b
            # Action activate function in current perceptron
            a = self.network[la].activate_function(z)
            out.append((z, a))

        # Back-propagation
        if train:
            deltas = []

            # Backward pass
            for layer in reversed(range(0, len(self.network))):
                # Reverse weighted sum and computed activation
                a = out[layer + 1][1]

                # Last layer
                if layer == len(self.network) - 1:
                    deltas.insert(0, self.least_squares_derivative(a, y) * self.derivative_activate_function(a))
                # Other layers that are not the last
                else:
                    deltas.insert(0, deltas[0] @ _W.T * self.derivative_activate_function(a))

                _W = self.network[layer].W

                # Gradiant descent for bias
                self.network[layer].b = self.network[layer].b - np.mean(deltas[0], axis=0,
                                                                        keepdims=True) * self.learning_rate

                # Gradiant descent for weights
                self.network[layer].W = self.network[layer].W - out[layer][1].T @ deltas[0] * self.learning_rate

        return out[-1][1]
