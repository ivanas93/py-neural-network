import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from sklearn.datasets import make_circles

# Build test data set using sklearn models
from NeuralNetwork import NeuralNetwork

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.08)
Y = Y[:, np.newaxis]

plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="blue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="red")
plt.axis("equal")
plt.show()

# Activation function (default = Sigmoid)
sigmoid = lambda x: 1 / (1 + np.e ** (-x))
sigmoid_derivative = lambda x: x * (1 - x)

# Neural network topology
topology = [p, 4, 8, 4, 1]

# Cost function
least_squares = lambda Yp, Yr: np.mean((Yp - Yr) ** 2)
least_squares_derivative = lambda Yp, Yr: (Yp - Yr)

# Learning rate (Gradiant descent)
learning_rate = 0.1
neural_network = NeuralNetwork(topology, sigmoid, sigmoid_derivative, least_squares, least_squares_derivative,
                               learning_rate)

# Execute epochs

# Save decrement cost function
loss = []

for i in range(5000):
    prediction_y = neural_network.train(X, Y)

    # Show graph every 10 interactions.
    if i % 100 == 0:
        # Save decrement
        loss.append(least_squares(prediction_y, Y))

        # Prepare plot
        res = 50
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = neural_network.train(np.array([[x0, x1]]), Y, train=False)[0][0]

        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="blue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="red")

        clear_output(wait=True)
        plt.show()
        plt.plot(range(len(loss)), loss)
        plt.show()
        time.sleep(0.5)
