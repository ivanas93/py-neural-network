from NeuralLayer import NeuralLayer as NeuralLayer


class NeuralNetwork:
    def __init__(self, topology, activate_function, derivative_activate_function):
        self.topology = topology
        self.activate_function = activate_function
        self.derivative_activate_function = derivative_activate_function
        self.network = self.create_network()

    def create_network(self):
        network = []
        for la, layer in enumerate(self.topology[:-1]):
            network.append(NeuralLayer(self.topology[1], self.topology[la + 1], self.activate_function,
                                       self.derivative_activate_function))
        return network

    def network(self):
        return self.network
