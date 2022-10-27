import numpy as np


def phi(x):

    return np.log(1.+np.exp(x))


def gen_input_output_data(n_in=30,
                          n_hidden=20,
                          n_out=10,
                          n_patterns=100000,
                          t_show_patterns=150,
                          t_offset=0,
                          weights=None):

    test_input = np.random.rand(n_patterns, n_in)

    if weights:
        W_10, W_21 = weights
    else:
        W_10 = 4.*(np.random.rand(n_hidden, n_in)-0.5)/np.sqrt(n_in)
        W_21 = 4.*(np.random.rand(n_out, n_hidden)-0.5) / \
            np.sqrt(n_hidden)

    test_output = (W_21 @ phi(W_10 @ test_input.T)).T

    t_ax = np.arange(n_patterns)*t_show_patterns + t_offset

    return t_ax, test_input, test_output, W_10, W_21
