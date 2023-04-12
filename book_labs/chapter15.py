import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from src.activation.relu import ReLU
from src.layer.dense import Dense
from src.layer.dropout import Dropout
from src.layer.softmax_classifier import SoftmaxClassifier
from src.optimizers.adam import Adam


# Training - Acc: 0.699, Loss: 0.714
# Validation - Acc: 0.767, Loss: 0.654
# Accuracy is greater than in training (But generally worse than without the dropout layer)
# Loss is a bit less than expected, this may sign overfitting
def ex_1():
    train_model(n_neurons=64)


# Training - Acc: 0.841, Loss: 0.493
# Validation - Acc: 0.840, Loss: 0.410
# Accuracy is the same as training (sign of overfitting)
# Same story as loss
def ex_2():
    train_model(n_neurons=512)


# Model trained with the Dropout layer
def train_model(n_neurons):
    X, y = spiral_data(samples=1000, classes=3)

    # Model
    dense_1 = Dense(2, n_neurons, l2_regularizer_w=5e-4, l2_regularizer_b=5e-4)  # 2 inputs (x, y coordinates), X outputs
    activation_1 = ReLU()
    dropout_1 = Dropout(0.1)
    dense_2 = Dense(n_neurons, 3)  # X inputs, 3 outputs (3 spiral classes)
    softmax_classifier = SoftmaxClassifier()
    optimizer = Adam(learning_rate=0.05, decay=5e-5)

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dropout_1.forward(activation_1.output)
        dense_2.forward(dropout_1.output)

        # Loss calculation
        data_loss = softmax_classifier.forward(dense_2.output, y)
        regularization_loss = \
            softmax_classifier.loss.get_regularization_loss(dense_1) + \
            softmax_classifier.loss.get_regularization_loss(dense_2)

        loss = data_loss + regularization_loss

        # stdout
        if epoch % 250 == 0:
            predictions = np.argmax(softmax_classifier.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, '
                  f'reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate}')

        # Backward pass
        softmax_classifier.backward(softmax_classifier.output, y)
        dense_2.backward(softmax_classifier.d_inputs)
        dropout_1.backward(dense_2.d_inputs)
        activation_1.backward(dropout_1.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.post_update_params()

    # Model validation
    X_test, y_test = spiral_data(samples=100, classes=3)

    # Forward pass
    dense_1.forward(X_test)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)

    # Loss
    loss = softmax_classifier.forward(dense_2.output, y_test)

    # Accuracy
    predictions = np.argmax(softmax_classifier.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    # stdout
    print(f'\nValidation - acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == '__main__':
    nnfs.init()
    ex_2()
