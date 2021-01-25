import numpy as np


class NeuralLayer:
    def __init__(self, n_connection, n_neural, activate_function, derivative_activate_function):
        self.activate_function = activate_function
        self.derivative_activate_function = derivative_activate_function
        self.b = np.random.rand(1, n_neural) * 2 - 1
        self.W = np.random.rand(n_connection, n_neural) * 2 - 1
