import nnfs
from nnfs.datasets import spiral_data
import numpy as np
from src.activation.softmax import Softmax
from src.activation.relu import ReLU
from src.loss.categorical_crossentropy import CategoricalCrossentropy
from src.layer.softmax_classifier import SoftmaxClassifier
from src.layer.dense import Dense


def ex_1():
    # Input & layer
    x = np.array([1.0, -2.0, 3.0])
    w = np.array([-3.0, -1.0, 2.0])
    b = 1.0

    # First pass
    forwarded = np.dot(x, w) + b
    relu = np.max(forwarded, 0)

    print(f"Pass before optimization: {relu}")

    '''
    End function looks like: relu(sum(inputs * weights) + bias)
        == relu(sum(x0 * w0, x1 * w1, x2 * w2, bias))
    '''

    # Backward pass
    # ...

    # The derivative from the next layer after relu
    d_value = 1.0

    # Derivative of relu(...) and the chain rule
    drelu_dforwarded = d_value * (1. if forwarded > 0 else 0)

    # ...

    # Derivative of a bias
    dsum_db = 1
    drelu_db = drelu_dforwarded * dsum_db

    # Derivatives of weight multiplication with respect to the input/weight
    drelu_dx0 = drelu_dforwarded * w[0]
    drelu_dx1 = drelu_dforwarded * w[1]
    drelu_dx2 = drelu_dforwarded * w[2]

    drelu_dw0 = drelu_dforwarded * x[0]
    drelu_dw1 = drelu_dforwarded * x[1]
    drelu_dw2 = drelu_dforwarded * x[2]

    # Gradients
    dx = np.array([drelu_dx0, drelu_dx1, drelu_dx2])
    dw = np.array([drelu_dw0, drelu_dw1, drelu_dw2])
    db = drelu_db

    # Optimize (Negative fraction to decrease the final output value)
    w += -0.001 * dw
    b += -0.001 * db

    # Second pass
    forwarded = np.dot(x, w) + b
    relu = np.max(forwarded, 0)

    print(f"Pass after optimization: {relu}")


def ex_2():
    weights = np.array([
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]).T

    # Passed in gradient from the next layer
    d_values = np.array([[1., 1., 1.]])

    # Gradient of the neuron function with respect to inputs
    d_inputs = np.dot(d_values[0], weights.T)

    print(d_inputs)


def ex_3():
    inputs = np.array([
        [1, 2, 3, 2.5],
        [2., 5., -1., 2],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    weights = np.array([
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]).T

    biases = np.array([[2, 3, 0.5]])

    output = np.array([
        [1, 2, -3, -4],
        [2, -7, -1, 3],
        [-1, 2, 5, -1]
    ])

    # Passed in gradient from the next layer
    d_values = np.array([
        [1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.],
    ])

    # Gradient of the neuron function with respect to inputs
    d_inputs = np.dot(d_values, weights.T)

    # Gradient of the neuron function with respect to weights
    d_weights = np.dot(inputs.T, d_values)

    # Derivative with respect for biases
    d_biases = np.sum(d_values, axis=0, keepdims=True)

    # Derivative of ReLU function
    d_values_for_relu = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    d_relu = d_values_for_relu.copy()
    d_relu[output <= 0] = 0


def ex_4():
    # Passed in gradient from the next layer
    d_values = np.array([
        [1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.]
    ])

    # 3 sets of inputs
    inputs = np.array([
        [1, 2, 3, 2.5],
        [2., 5., -1., 2],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    # 3 sets of weights - one for each neuron
    # 4 inputs, therefore 4 weights
    # Must be transposed
    weights = np.array([
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]).T

    # One bias for each neuron
    biases = np.array([[2, 3, 0.5]])

    # Forward pass
    layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
    relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation

    # Optimization and brackpropagation
    # ReLU activation
    d_relu = relu_outputs.copy()
    d_relu[layer_outputs <= 0] = 0

    # Dense layer
    # d_inputs - multiply by weights
    d_inputs = np.dot(d_relu, weights.T)

    # d_weights - multiply by inputs
    d_weights = np.dot(inputs.T, d_relu)

    # d_biases - sum values
    d_biases = np.sum(d_relu, axis=0, keepdims=True)

    # Update paramteres
    weights += -0.001 * d_weights
    biases += -0.001 * d_biases

    print(weights)
    print(biases)


def ex_5():
    softmax_output = [0.7, 0.1, 0.2]
    softmax_output = np.array(softmax_output).reshape(-1, 1)

    jacobian_matrix = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
    print(jacobian_matrix)


def ex_6():
    softmax_outputs = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ])

    y = np.array([0, 1, 1])

    softmax_classifier = SoftmaxClassifier()
    softmax_classifier.backward(softmax_outputs, y)
    d_values = softmax_classifier.d_inputs

    activation = Softmax()
    activation.output = softmax_outputs

    loss = CategoricalCrossentropy()
    loss.backward(softmax_outputs, y)
    activation.backward(loss.d_inputs)
    d_values_2 = activation.d_inputs

    print(f'Gradients of combined loss and activation:\n{d_values}\n')
    print('Gradients of separated loss and activation:\n', d_values_2)


def ex_7():
    X, y = spiral_data(100, 3)

    # Model
    dense_1 = Dense(2, 64)
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)
    softmax_classifier = SoftmaxClassifier()

    # Forward pass
    dense_1.forward(X)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)
    loss = softmax_classifier.forward(dense_2.output, y)

    # stdout - model
    print(softmax_classifier.output[:5])
    print(f'loss: {loss}')

    # Accuracy
    predictions = np.argmax(softmax_classifier.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print(f'acc: {accuracy}\n')

    # Backward pass
    softmax_classifier.backward(softmax_classifier.output, y)
    dense_2.backward(softmax_classifier.d_inputs)
    activation_1.backward(dense_2.d_inputs)
    dense_1.backward(activation_1.d_inputs)

    # stdout - gradients
    print(dense_1.d_weights)
    print(dense_1.d_biases)
    print(dense_2.d_weights)
    print(dense_2.d_biases)


if __name__ == '__main__':
    nnfs.init()
    ex_7()
