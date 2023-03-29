from p1 import calc_neuron_output


def ex_1():
    inputs = [1, 2, 3, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1.0]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    output = [
        calc_neuron_output(inputs, weights1, bias1),
        calc_neuron_output(inputs, weights2, bias2),
        calc_neuron_output(inputs, weights3, bias3),
    ]

    print(output)


""" Layer of 3 neurons """
if __name__ == '__main__':
    ex_1()
