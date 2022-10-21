from layer import Layer
import numpy as np
from scipy import special


class SoftmaxCE:
    def __init__(self):
        self.output = None
        self.y_actual = None
        self.y_pred = None
        self.batch_size = None

    def forward(self, input, y_actual):
        self.output = 0
        self.y_actual = y_actual
        self.batch_size =len(input)
        e_x = np.exp(input - np.max(input))
        self.y_pred = (e_x / np.sum(e_x)) + 1e-15
        self.output += - np.sum(special.xlogy(self.y_actual, self.y_pred))
        self.output /= len(input) # Divide by batch size
        return self.output

    def backward(self):
        return self.y_pred - self.y_actual
        # n = np.size(self.output)
        # return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
