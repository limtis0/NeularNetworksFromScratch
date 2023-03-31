import nnfs
from nnfs.datasets import vertical_data, spiral_data
import numpy as np
from l4 import LayerDense
from l5 import ActivationReLU
from l6 import ActivationSoftmax
from l8 import CategoricalCrossentropyLoss


# Randomly tweaking best weights and biases
def ex_1():
    X, y = vertical_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)  # (x, y) input
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)  # One of 3 classes as an output
    activation2 = ActivationSoftmax()

    loss_function = CategoricalCrossentropyLoss()

    lowest_loss = float('inf')
    best_dense_1_weights = dense1.weights.copy()
    best_dense_1_biases = dense1.biases.copy()
    best_dense_2_weights = dense2.weights.copy()
    best_dense_2_biases = dense2.biases.copy()

    for i in range(100_000):
        # Adjusting
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        # Predicting
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)

        # Accuracy
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if loss < lowest_loss:
            print(f'New set of weights found: iteration {i}, loss {loss}, acc {accuracy*100}%')

            best_dense_1_weights = dense1.weights.copy()
            best_dense_1_biases = dense1.biases.copy()
            best_dense_2_weights = dense2.weights.copy()
            best_dense_2_biases = dense2.biases.copy()

            lowest_loss = loss
        else:
            dense1.weights = best_dense_1_weights.copy()
            dense1.biases = best_dense_1_biases.copy()
            dense2.weights = best_dense_2_weights.copy()
            dense2.biases = best_dense_2_biases.copy()


if __name__ == '__main__':
    nnfs.init()
    ex_1()
