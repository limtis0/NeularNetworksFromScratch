import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from src.optimizers.optimizer import Optimizer
from src.optimizers.sgd import SGD
from src.optimizers.adagrad import AdaGrad
from src.optimizers.rmsprop import RMSProp
from src.optimizers.adam import Adam

from src.activation.relu import ReLU
from src.layer.softmax_classifier import SoftmaxClassifier
from src.layer.dense import Dense


# Learning rate == 1
def ex_1():
    train_model(SGD())


# With addded decay
def ex_2():
    train_model(SGD(learning_rate=1, decay=1e-3))


# With addded momentum
def ex_3():
    train_model(SGD(learning_rate=1, decay=1e-3, momentum=0.5))


# With better config values
# The model is achieving 0.099 loss and 96.7% accuracy in just a 10000 epochs
def ex_4():
    train_model(SGD(learning_rate=1.1, decay=1e-3, momentum=0.9))


# AdaGrad optimizer, works better with smaller decay
# For this dataset, it shows a bit worse than SGD
def ex_5():
    train_model(AdaGrad(learning_rate=1.02, decay=1e-5))


# Root Mean Square Propagation
def ex_6():
    train_model(RMSProp(learning_rate=0.005, decay=1e-5, rho=0.999))


# Adam optimizer
# Able to get to 0.983 accuracy and 0.054 loss
def ex_7():
    train_model(Adam(learning_rate=0.06, decay=8e-6))


def train_model(optimizer: Optimizer):
    X, y = spiral_data(samples=100, classes=3)

    # Model
    dense_1 = Dense(2, 64)  # x,y coordinate input, 64 output
    activation_1 = ReLU()
    dense_2 = Dense(64, 3)  # 64 inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()

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

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

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
    ex_7()
