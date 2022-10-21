from layer import Layer
import numpy as np


# Assumed square matrices. Therefore, filter_height and filter_width
# or input_height and input_width can be used interchangeably.
class Pooling(Layer):
    def __init__(self, input_shape, filter_shape, stride):  # Channels is first parameter of filter size
        # Unpacking
        super().__init__()
        self.input_batch, self.input_depth, self.input_height, self.input_width = input_shape
        self.filter_depth, self.filter_height, self.filter_width = filter_shape
        self.stride = stride

        # Self.shapes
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.output_shape = (self.input_batch, self.input_depth, int((self.input_height - self.filter_height) / 2) + 1,
                             int((self.input_height - self.filter_height) / 2) + 1)

        # Randomized initialization
        self.output = np.zeros(self.output_shape)
        self.filter = np.random.randn(self.filter_depth, self.filter_height, self.filter_width)
        self.cache = np.zeros((self.input_batch, self.input_depth, self.input_height, self.input_height))
        self.slide_times = None
        self.grad_pooling= None

    def forward(self, input):
        self.input = input
        self.slide_times = int((self.input_height - self.filter_height) / self.stride) + 1

        for i in range(0, self.input_batch):
            for j in range(0, self.input_depth):
                for r in range(0, self.slide_times):
                    for c in range(0, self.slide_times):
                        row_start = r * self.stride
                        row_end = row_start + self.filter_height
                        column_start = c * self.stride
                        column_end = column_start + self.filter_height

                        input_slice = self.input[i, j, row_start: row_end, column_start:column_end]
                        self.output[i, j, r, c] = np.max(input_slice)

                        # Caching for backpropagation. Location of the max values are stored in binary table.
                        truth_table = 1 * (input_slice == np.max(input_slice))
                        self.cache[i, j, row_start: row_end, column_start:column_end] += truth_table

        return self.output

        # grad_pool = np.random.randint(1, 4, size=(depth, int((self.input_height - filter_size) / stride) + 1, int((self.input_height - filter_size) / stride) + 1))  # Same as output shape

    def backward(self, output_grad, learning_rate):
        for i in range(0, self.input_depth):
            for r in range(0, self.slide_times):
                for c in range(0, self.slide_times):
                    row_start = r * self.stride
                    row_end = row_start + self.filter_height
                    column_start = c * self.stride
                    column_end = column_start + self.filter_height
                    self.cache[i, row_start: row_end, column_start:column_end] \
                        = self.cache[i, row_start: row_end, column_start:column_end] * output_grad[i, r, c]

        self.grad_pooling = self.cache
        return self.grad_pooling

