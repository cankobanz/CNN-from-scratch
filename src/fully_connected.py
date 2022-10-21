import numpy as np
from layer import Layer


class FCLayer(Layer):
    def __init__(self, input_neuron_size, output_neuron_size):
        super().__init__()
        self.output_neuron_size = output_neuron_size
        self.weights = np.random.randn(output_neuron_size, input_neuron_size)
        self.bias = np.random.randn(output_neuron_size, 1)

    # Simple forward propagation method is implemented.
    def forward(self, input):
        self.input = input
        batch_size = len(self.input)
        self.output = np.empty((batch_size,self.output_neuron_size, 1))
        for i in range(0, len(self.input)):
            self.output[i] = np.dot(self.weights, self.input[i]) + self.bias
        return self.output

    # gradient_y = dE/dY which is necessary for previous layer.
    # gradient_x= dE/dX
    # gradient_w= dE/dW
    # gradient_b= dE/dB
    def backward(self, gradient_y, learning_rate):
        gradient_x = np.dot(np.transpose(self.weights), gradient_y)
        gradient_w = np.dot(gradient_y, np.transpose(self.input))
        gradient_b = gradient_y

        # after gradients are calculated, weight and bias updated.
        # gradient_x is returned because latter layer's gradient_x is equals to gradient_y of former layer.
        self.weights = self.weights - learning_rate * gradient_w
        self.bias = self.bias - learning_rate * gradient_b
        return gradient_x