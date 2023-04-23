import numpy as np


def phi(x):
    return np.maximum(0.0, x)


def act_func_input(x):
    return 3.5*np.log(1.0+np.exp((x-0.2)*15.0))/15.0


def yinyang(x, center=np.zeros(2), R=1.):
    R_med = 0.5 * R
    R_small = 0.5 * R_med

    c_l = np.array([-R_med, 0.])
    c_r = np.array([R_med, 0.])

    d = np.linalg.norm(x - center, axis=-1)

    d_l = np.linalg.norm(x - c_l - center, axis=-1)
    d_r = np.linalg.norm(x - c_r - center, axis=-1)

    c0 = np.logical_or((d_l <= R_small), (d_r <= R_small))

    c1 = np.logical_or(d_l <= R_med, x[..., 1] >= 0.)
    c1 = np.logical_and(c1, d <= R)
    c1 = np.logical_and(c1, d_r > R_med)
    c1 = np.logical_and(c1, np.logical_not(c0))

    c2 = np.logical_and(d <= R, np.logical_not(c0))
    c2 = np.logical_and(c2, np.logical_not(c1))

    cout = (d > R)

    return np.stack([cout, c0, c1, c2], axis=-1).astype("float")


def gen_output_data(input_data):

    output_data = yinyang(input_data, center=np.ones(2)*0.5, R=0.5)

    return output_data
