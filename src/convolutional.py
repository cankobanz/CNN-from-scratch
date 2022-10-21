import numpy as np
from layer import Layer
from scipy import signal


# Assumed square matrices. Therefore, filter_height and filter_width
# or input_height and input_width can be used interchangeably.
class Convolutional(Layer):
    def __init__(self, input_shape, filter_size):  # Channels is first parameter of filter size
        # Unpacking
        super().__init__()
        self.input_batch, self.input_depth, self.input_height, self.input_width = input_shape
        self.channels, self.filter_depth, self.filter_height, self.filter_width = filter_size

        # Self.shapes
        self.input_shape = input_shape
        self.filter_shape = filter_size
        self.output_shape = (self.input_batch, self.channels, self.input_height - self.filter_height + 1,
                             self.input_height - self.filter_height + 1)

        # Randomized initialization
        self.output = np.empty(self.output_shape)
        self.filter = np.random.randn(self.channels, self.filter_depth, self.filter_height, self.filter_width)
        self.bias = np.random.randn(self.channels, self.input_height - self.filter_height + 1,
                                    self.input_height - self.filter_height + 1)  # Shape is same as output_shape

    def forward(self, input):
        self.input = input

        slide_times = self.input_height - self.filter_height + 1
        # TODO: Buraya batch için bir loop daha gelebilir

        for i in range(0, self.input_batch):
            for j in range(0, self.channels):
                for r in range(0, slide_times):
                    for c in range(0, slide_times):
                        input_slice = self.input[i, :, r: r + self.filter_height, c:c + self.filter_height]
                        self.output[i, j, r, c] = self.single_conv(input_slice, self.filter[j], self.bias[j, r, c])

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.filter_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.channels):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.filter[i, j], "full")

        self.filter -= learning_rate * kernels_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

    @staticmethod
    def single_conv(input_slice, f, bias):
        if input_slice.shape != f.shape:
            raise Exception("During single_conv(), dimensions don't match.")

        def matrix_sum(x):
            for i in range(0, x.ndim):
                x = sum(x)
            return x

        return matrix_sum(np.multiply(input_slice, f)) + bias


    # def backward(self, gradient_y, learning_rate):
    #     gradient_f = np.empty(self.filter_shape)
    #     gradient_y_channels, gradient_y_depth, gradient_y_height, gradient_y_width = gradient_y.shape  # Acts like filter.
    #
    #     slide_times = self.input_height - gradient_y_height + 1  # output_shape[2]
    #     for j in range(0, gradient_y_channels):
    #         for i in range(0,gradient_y_depth):
    #             for r in range(0, slide_times):
    #                 for c in range(0, slide_times):
    #                     input_slice = self.input[:, r: r + gradient_y_height, c:c + gradient_y_width]
    #                     gradient_f[j, i, r, c] = self.single_conv(input_slice, gradient_y[j], 0) # There is not bias so it is taken as 0
    #
    #     return gradient_f