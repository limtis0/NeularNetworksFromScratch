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
    weights = np.array([
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]).T

    # Passed in gradient from the next layer
    d_values = np.array([
        [1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.],
    ])

    # Gradient of the neuron function with respect to inputs
    d_inputs = np.dot(d_values, weights.T)

    print(d_inputs)


if __name__ == '__main__':
    nnfs.init()
    ex_3()
