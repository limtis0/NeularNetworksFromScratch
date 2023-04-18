import copy
import pickle

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
    def __init__(self, *layers, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.input_layer = Input()
        self.layers = [layer for layer in layers]
        self.trainable_layers = [layer for layer in layers if isinstance(layer, Dense)]
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

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.initialize(y)

        # Calculate amount of steps
        training_steps = self._calculate_steps(X, batch_size)

        # Training
        for epoch in range(1, epochs + 1):
            print(f'--- Epoch {epoch} ---')

            self.loss.reset_accumulated()
            self.accuracy.reset_accumulated()

            for step in range(training_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)
                data_loss = self.loss.calculate(output, batch_y)

                # Step stdout
                if step % print_every == 0 or step == training_steps - 1:
                    regularization_loss = self.loss.get_regularization_loss(*self.trainable_layers)
                    loss = data_loss + regularization_loss

                    predictions = self.output_layer.get_predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)

                    print(f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, '
                          f'reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            # Epoch stdout
            epoch_data_loss = self.loss.calculate_accumulated()
            epoch_regularization_loss = self.loss.get_regularization_loss(*self.trainable_layers)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'Epoch {epoch} finished - acc: {epoch_accuracy:.3f}, loss: {epoch_loss} '
                  f'(data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), '
                  f'lr: {self.optimizer.current_learning_rate}\n')

        # Validation
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size)

    def evaluate(self, X_val, y_val, batch_size=None):
        self.loss.reset_accumulated()
        self.accuracy.reset_accumulated()

        steps_val = self._calculate_steps(X_val, batch_size)

        for step in range(steps_val):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)

            predictions = self.output_layer.get_predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'Evaluation - acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

    @staticmethod
    def _calculate_steps(X, batch_size: int):
        # Calculate the amount of steps_train per epoch using batch_size
        if batch_size is None:
            return 1

        steps_train = len(X) // batch_size
        if steps_train * batch_size < len(X):
            steps_train += 1

        return steps_train

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

    def save(self, path):
        model = copy.deepcopy(self)

        # Cleanup
        model.loss.reset_accumulated()
        model.accuracy.reset_accumulated()

        del model.input_layer.output
        del model.loss.d_inputs

        for layer in model.layers:
            try:
                del layer.inputs
                del layer.output
                del layer.d_inputs
                del layer.d_weights
                del layer.d_biases

            except AttributeError:  # Not all of the layers have d_weights/d_biases values
                pass

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
