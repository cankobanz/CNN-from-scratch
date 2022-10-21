import layer
import numpy as np
Layer = layer.Layer


class ReLU(Layer):
    # Forward takes relu of the incoming scalar, vector or matrix and returns output of layer.
    def forward(self, input):
        self.input = input
        self.output = self.relu(self.input)
        return self.output

    # Backward takes output gradient which is coming from next layer for back propagation.
    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, self.relu_derivative(self.input))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return 1. * (x >= 0)

