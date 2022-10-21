from layer import Layer
import numpy as np
from scipy import special


class SoftmaxCE:
    def __init__(self):
        self.output = None
        self.y_actual = None
        self.y_pred = None

    def forward(self, input, y_actual):
        self.y_actual = y_actual
        e_x = np.exp(input - np.max(input))
        self.y_pred = (e_x / np.sum(e_x)) + 1e-15
        self.output = - np.sum(special.xlogy(self.y_actual, self.y_pred))
        return self.output

    def backward(self):
        return self.y_pred - self.y_actual
