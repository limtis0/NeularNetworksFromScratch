import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
from src.activation.relu import ReLU
from src.activation.linear import Linear
from src.layer.dense import Dense
from src.loss.mean_squared_error import MeanSquaredError
from src.optimizers.adam import Adam


# Model trained with the Dropout layer
def train_model():
    X, y = sine_data()

    # Model
    dense_1 = Dense(1, 64)  # 2 inputs (x, y coordinates), X outputs
    activation_1 = ReLU()
    dense_2 = Dense(64, 64)  # X inputs, 3 outputs (3 spiral classes)
    activation_2 = ReLU()
    dense_3 = Dense(64, 1)
    activation_3 = Linear()
    loss_function = MeanSquaredError()
    optimizer = Adam(learning_rate=0.005, decay=1e-3)

    accuracy_precision = np.std(y) / 250

    for epoch in range(10_001):
        # Forward pass
        dense_1.forward(X)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)
        dense_3.forward(activation_2.output)
        activation_3.forward(dense_3.output)

        # Loss calculation
        data_loss = loss_function.calculate(activation_3.output, y)
        regularization_loss = \
            loss_function.get_regularization_loss(dense_1) + \
            loss_function.get_regularization_loss(dense_2) + \
            loss_function.get_regularization_loss(dense_3)

        loss = data_loss + regularization_loss

        # stdout
        if epoch % 250 == 0:
            predictions = activation_3.output
            accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, '
                  f'reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_function.backward(activation_3.output, y)
        activation_3.backward(loss_function.d_inputs)
        dense_3.backward(activation_3.d_inputs)
        activation_2.backward(dense_3.d_inputs)
        dense_2.backward(activation_2.d_inputs)
        activation_1.backward(dense_2.d_inputs)
        dense_1.backward(activation_1.d_inputs)

        # Optimize
        optimizer.pre_update_params()
        optimizer.update_params(dense_1)
        optimizer.update_params(dense_2)
        optimizer.update_params(dense_3)
        optimizer.post_update_params()

    # Model validation
    X_test, y_test = sine_data()

    # Forward pass
    dense_1.forward(X_test)
    activation_1.forward(dense_1.output)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)
    dense_3.forward(activation_2.output)
    activation_3.forward(dense_3.output)

    plt.plot(X_test, y_test)
    plt.plot(X_test, activation_3.output)
    plt.show()

    # Loss
    loss = loss_function.calculate(activation_3.output, y_test)

    # Accuracy
    predictions = activation_3.output
    accuracy = np.mean(np.absolute(predictions - y_test) < accuracy_precision)

    # stdout
    print(f'\nValidation - acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == '__main__':
    nnfs.init()
    train_model()
