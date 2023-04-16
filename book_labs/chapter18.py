import nnfs
from nnfs.datasets import sine_data, spiral_data

from src.activation.relu import ReLU
from src.activation.linear import Linear
from src.activation.softmax import Softmax

from src.layer.dense import Dense
from src.layer.dropout import Dropout

from src.model.model import Model

from src.optimizers.adam import Adam

from src.loss.mean_squared_error import MeanSquaredError
from src.loss.categorical_crossentropy import CategoricalCrossentropy

from src.accuracy.regression import Regression
from src.accuracy.categorical import Categorical


def train_model_regression():
    X, y = sine_data()
    X_test, y_test = sine_data()

    # Model
    model = Model(
        Dense(1, 64),
        ReLU(),
        Dense(64, 64),
        ReLU(),
        Dense(64, 1),
        Linear(),
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.005, decay=1e-3),
        accuracy=Regression()
    )

    model.train(X, y, epochs=10_000, print_every=100, validation_data=(X_test, y_test))


def train_model_categorical():
    X, y = spiral_data(samples=1000, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)

    model = Model(
        Dense(2, 512, l2_regularizer_w=5e-4, l2_regularizer_b=5e-4),
        ReLU(),
        Dropout(rate=0.1),
        Dense(512, 3),
        Softmax(),
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Categorical()
    )

    model.train(X, y, epochs=10_000, print_every=100, validation_data=(X_test, y_test))


if __name__ == '__main__':
    nnfs.init()
    train_model_categorical()
