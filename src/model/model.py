import numpy as np

from src.model.input import Input
from src.layer.dense import Dense

from src.activation.softmax import Softmax
from src.loss.categorical_crossentropy import CategoricalCrossentropy
from src.layer.softmax_classifier import SoftmaxClassifier

from src.activation.activation import Activation
from src.loss.loss import Loss
from src.optimizers.optimizer import Optimizer
from src.accuracy.accuracy import Accuracy

from typing import Optional


class Model:
    def __init__(self, *args, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.input_layer = Input()
        self.layers = [layer for layer in args]
        self.trainable_layers = [layer for layer in args if isinstance(layer, Dense)]
        self.output_layer: Activation = self.layers[-1]

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

        # Softmax + Categorical Cross-Entropy optimization
        self.softmax_classifier: Optional[SoftmaxClassifier] = None

        self._set_layer_order()

    def _set_layer_order(self):
        layer_count = len(self.layers)

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

        # Softmax + Categorical Cross-Entropy optimization
        if isinstance(self.output_layer, Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier = SoftmaxClassifier()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.initialize(y)

        # Training
        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)
            data_loss = self.loss.calculate(output, y)

            # stdout
            if epoch % print_every == 0:
                regularization_loss = self.loss.get_regularization_loss(*self.trainable_layers)
                loss = data_loss + regularization_loss

                predictions = self.output_layer.get_predictions(output)
                accuracy = self.accuracy.calculate(predictions, y)

                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, '
                      f'reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

        # Validation
        if validation_data is not None:
            X_val, y_val = validation_data

            output = self.forward(X_val, training=False)
            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer.get_predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f'\nValidation - acc: {accuracy:.3f}, loss: {loss:.3f}')

    def forward(self, X, training: bool):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output, training=training)

        return self.layers[-1].output

    def backward(self, output, y):
        # Softmax + Categorical Cross-Entropy optimization
        if self.softmax_classifier:
            return self._backward_softmax_classifier(output, y)

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)

    # Softmax + Categorical Cross-Entropy optimization
    def _backward_softmax_classifier(self, output, y):
        self.softmax_classifier.backward(output, y)
        self.layers[-1].d_inputs = self.softmax_classifier.d_inputs

        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.d_inputs)
