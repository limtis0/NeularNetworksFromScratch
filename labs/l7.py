import nnfs
import numpy as np


# Categorical cross-entropy
def ex_1():
    softmax_output = [0.7, 0.1, 0.2]
    target_output = [1, 0, 0]

    loss = -np.log(softmax_output[target_output == 1])
    print(loss)


if __name__ == '__main__':
    nnfs.init()
    ex_1()
