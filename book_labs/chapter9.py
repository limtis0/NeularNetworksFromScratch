import nnfs
import numpy as np


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


if __name__ == '__main__':
    nnfs.init()
    ex_4()
