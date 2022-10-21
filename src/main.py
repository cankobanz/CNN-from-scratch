import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from fully_connected import FCLayer
from convolutional import Convolutional
from flatten import Flatten
from activations import Sigmoid
from softmax_cross_entropy import SoftmaxCE

from relu import ReLU
from pooling import Pooling
from linear import Linear


def preprocess_data(x, y, limit_get):
    zero_index = np.where(y == 0)[0][:limit_get]
    one_index = np.where(y == 1)[0][:limit_get]
    two_index = np.where(y == 2)[0][:limit_get]
    three_index = np.where(y == 3)[0][:limit_get]
    four_index = np.where(y == 4)[0][:limit_get]
    five_index = np.where(y == 5)[0][:limit_get]
    six_index = np.where(y == 6)[0][:limit_get]
    seven_index = np.where(y == 7)[0][:limit_get]
    eight_index = np.where(y == 8)[0][:limit_get]
    nine_index = np.where(y == 9)[0][:limit_get]
    all_indices = np.hstack((zero_index, one_index, two_index, three_index, four_index, five_index, six_index, seven_index, eight_index, nine_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

"""""
You can CHANGE THIS PART.
"""""
limit = 100 # Number of data will be loaded is limit*10 for training
epochs = 20
learning_rate = 0.1
verbose = True
"""""
You can CHANGE THIS PART.
"""""

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, limit)
x_test, y_test = preprocess_data(x_test, y_test, limit)

# Working network structure. To see correct structure comment this and uncomment following.
network = [
    Convolutional(input_shape=(1, 28, 28), filter_size=(4, 1, 5, 5)),
    ReLU(),
    Pooling(input_shape=(4, 24, 24), filter_shape=(4, 2, 2), stride=2),
    Flatten((4, 12, 12), (4 * 12 * 12, 1)),
    FCLayer(4 * 12 * 12, 100),
    Sigmoid(),
    FCLayer(100, 10),
    Sigmoid()
]

# Unworking network structure. To see correct structure uncomment this and uncomment previous.

# network = [
#     Convolutional(input_shape=(1, 28, 28), filter_size=(4, 1, 5, 5)),
#     ReLU(),
#     Pooling(input_shape=(4, 24, 24), filter_shape=(4, 2, 2), stride=2),
#     Convolutional(input_shape=(4, 12, 12), filter_size=(8, 4, 5, 5)),
#     ReLU(),
#     Pooling(input_shape=(8, 8, 8), filter_shape=(8, 2, 2), stride=2),
#     Flatten((8, 4, 4), (8 * 4 * 4, 1)),
#     FCLayer(128, 128),
#     ReLU(),
#     FCLayer(128, 10),
#     Linear()
# ]

#Training part.
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        softmax_ce = SoftmaxCE()

        # error
        error += softmax_ce.forward(output, y)

        grad = softmax_ce.backward()
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(x_train)
    if verbose:
        print(f"{e + 1}/{epochs}, error={error}")

correct = 0
total = 0
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    # print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if np.argmax(output) == np.argmax(y):
        correct += 1
    total += 1

accuracy = correct/total
print(accuracy)
