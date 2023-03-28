from typing import List


def calc_neuron_output(input_values: List[float], weight_values: List[float], bias_value: float):
    return sum(input_values[i] * weight_values[i] for i in range(len(input_values))) + bias_value


if __name__ == '__main__':
    # Randomly chosen data
    inputs = [1.2, 5.1, 2.1]
    weights = [3.1, 2.1, 8.7]
    bias = 3

    print(calc_neuron_output(inputs, weights, bias))
