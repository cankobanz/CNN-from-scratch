import layer
import numpy as np

Layer = layer.Layer


class Linear(Layer):
    def forward(self, input):
        self.input = input
        self.output = self.linear(self.input)
        return self.output

    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, self.linear_derivative(self.input))

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        if type(x) == int or float:
            return 1
        else:
            return np.ones(x.shape)
