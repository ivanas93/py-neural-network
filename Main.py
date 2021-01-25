import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

# Build test data set using sklearn models
from NeuralNetwork import NeuralNetwork

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.08)

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="blue")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="red")
plt.axis("equal")
plt.show()

# Activation function (default = Sigmoid)
sigmoid = lambda x: 1 / (1 + np.e ** (-x))
sigmoid_derivate = lambda x: x * (1 - x)

# Neural network topology
topology = [p, 4, 8, 16, 8, 4, 1]

NeuralNetwork(topology, sigmoid, sigmoid_derivate)

print(NeuralNetwork.network)
