import numpy as np
from typing import List
from l1 import calc_neuron_output


# Lab2 in a function
def calc_layer_output(input_values: List[float], weight_values: List[List[float]], bias_values: List[float]):
    return [calc_neuron_output(input_values, w, b) for w, b in zip(weight_values, bias_values)]


# Calculating a layer
def ex_1():
    inputs = [1, 2, 3, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2, 3, 0.5]

    print(calc_layer_output(inputs, weights, biases))


# Using numpy instead of plain Python
def ex_2():
    inputs = [1, 2, 3, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2, 3, 0.5]

    output = np.dot(weights, inputs) + biases

    print(output)


if __name__ == '__main__':
    ex_2()
