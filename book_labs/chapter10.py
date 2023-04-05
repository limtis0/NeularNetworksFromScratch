import nnfs
from nnfs.datasets import spiral_data
import numpy as np
from src.optimizers.sgd import SGD
from src.activation.relu import ReLU
from src.layer.softmax_classifier import SoftmaxClassifier
from src.layer.dense import Dense


# Learning rate == 1
def ex_1():
    X, y = spiral_data(samples=100, classes=3)

    # Model
    dense_1 = Dense(2, 64)  # x,y coordinate input, 64 output
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)  # 64 inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()
    optimizer = SGD()

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        loss = softmax_classifier.forward(dense_2.output, y)

        # stdout
        if epoch % 100 == 0:
            predictions = np.argmax(softmax_classifier.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')

        # Backward pass
        softmax_classifier.backward(softmax_classifier.output, y)
        dense_2.backward(softmax_classifier.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)


# With addded decay
def ex_2():
    X, y = spiral_data(samples=100, classes=3)

    # Model
    dense_1 = Dense(2, 64)  # x,y coordinate input, 64 output
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)  # 64 inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()
    optimizer = SGD(learning_rate=1, decay=1e-3)

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        loss = softmax_classifier.forward(dense_2.output, y)

        # stdout
        if epoch % 100 == 0:
            predictions = np.argmax(softmax_classifier.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learing_rate}')

        # Backward pass
        softmax_classifier.backward(softmax_classifier.output, y)
        dense_2.backward(softmax_classifier.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.post_update_params()


# With addded momentum
def ex_3():
    X, y = spiral_data(samples=100, classes=3)

    # Model
    dense_1 = Dense(2, 64)  # x,y coordinate input, 64 output
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)  # 64 inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()
    optimizer = SGD(learning_rate=1, decay=1e-3, momentum=0.5)

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        loss = softmax_classifier.forward(dense_2.output, y)

        # stdout
        if epoch % 100 == 0:
            predictions = np.argmax(softmax_classifier.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learing_rate}')

        # Backward pass
        softmax_classifier.backward(softmax_classifier.output, y)
        dense_2.backward(softmax_classifier.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.post_update_params()


# With better config values
# The model is achieving 0.099 loss and 96.7% accuracy in just a 10000 epochs
def ex_4():
    X, y = spiral_data(samples=100, classes=3)

    # Model
    dense_1 = Dense(2, 64)  # x,y coordinate input, 64 output
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)  # 64 inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()
    optimizer = SGD(learning_rate=1.1, decay=1e-3, momentum=0.9)

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        loss = softmax_classifier.forward(dense_2.output, y)

        # stdout
        if epoch % 500 == 0:
            predictions = np.argmax(softmax_classifier.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learing_rate}')

        # Backward pass
        softmax_classifier.backward(softmax_classifier.output, y)
        dense_2.backward(softmax_classifier.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.post_update_params()


if __name__ == '__main__':
    nnfs.init()
    ex_4()
