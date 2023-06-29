import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def tanh(x): return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))
def tanh_derivative(x): return (1 + tanh(x)) * (1 - tanh(x))

class NeuralNetwork:
    def __init__(self, net_arch):
        self.activation_func = tanh
        self.activation_derivative = tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1000
        self.net_arch = net_arch

        # Init the weights with random values in the range (-1, 1)
        self.weights = []
        for layer in range(len(net_arch - 1)):
            w = 2 * np.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1
            self.weights.append(w)

    def fit(self, data, labels, learning_rate = 0.1, epochs=10):
        return