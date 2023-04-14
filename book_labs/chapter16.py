import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from src.activation.relu import ReLU
from src.activation.sigmoid import Sigmoid
from src.layer.dense import Dense
from src.loss.binary_crossentropy import BinaryCrossentropy
from src.optimizers.adam import Adam


# Model trained with the Dropout layer
def train_model():
    X, y = spiral_data(samples=100, classes=2)
    y = y.reshape(-1, 1)

    # Model
    dense_1 = Dense(2, 64, l2_regularizer_w=5e-4, l2_regularizer_b=5e-4)  # 2 inputs (x, y coordinates), X outputs
    activation_1 = ReLU()
    dense_2 = Dense(64, 1)  # X inputs, 3 outputs (3 spiral classes)
    activation_2 = Sigmoid()
    loss_function = BinaryCrossentropy()
    optimizer = Adam(decay=5e-7)

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)

        # Loss calculation
        data_loss = loss_function.calculate(activation_2.output, y)
        regularization_loss = \
            loss_function.get_regularization_loss(dense_1) + \
            loss_function.get_regularization_loss(dense_2)

        loss = data_loss + regularization_loss

        # stdout
        if epoch % 250 == 0:
            predictions = (activation_2.output > 0.5) * 1
            accuracy = np.mean(predictions == y)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, '
                  f'reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_function.backward(activation_2.output, y)
        activation_2.backward(loss_function.d_inputs)
        dense_2.backward(activation_2.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.post_update_params()

    # Model validation
    X_test, y_test = spiral_data(samples=100, classes=2)
    y_test = y_test.reshape(-1, 1)

    # Forward pass
    dense_1.forward(X_test)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)

    # Loss
    loss = loss_function.calculate(activation_2.output, y_test)

    # Accuracy
    predictions = (activation_2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_test)

    # stdout
    print(f'\nValidation - acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == '__main__':
    nnfs.init()
    train_model()
