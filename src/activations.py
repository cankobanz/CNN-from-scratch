import numpy as np
from layer import Layer
from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_derivative(x):
            return np.ones(x.shape)

        super().__init__(linear, linear_derivative)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_derivative(x):
            return 1. * (x >= 0)

        super().__init__(relu, relu_derivative)


