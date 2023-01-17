import numpy as np

def phi(x):
    #return 3.5 * np.maximum(0., x - 0.2)
    return 3.5*np.log(1.0+np.exp((x-0.2)*15.0))/15.0

def gen_output_data(weights, input_data):

    w_10, w_21 = weights

    hidden_data = np.tensordot(w_10, phi(input_data).T, axes=([-1],[0])).T
    output_data = np.tensordot(w_21, phi(hidden_data).T, axes=([-1], [0])).T

    return hidden_data, output_data