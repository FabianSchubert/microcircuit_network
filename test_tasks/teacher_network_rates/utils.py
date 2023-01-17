import numpy as np


def phi(x):
    #return x
    return np.log(1.+np.exp(x))


def gen_output_data(weights, input_data):

    w_10, w_21 = weights

    hidden_data = np.tensordot(w_10, input_data.T, axes=([-1],[0])).T
    output_data = np.tensordot(w_21, phi(hidden_data).T, axes=([-1], [0])).T

    return hidden_data, output_data
