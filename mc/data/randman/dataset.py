import numpy as np
from numba import jit


@jit
def f(x, th_a, th_b, th_c, alpha, m, d, n_c):
    s = x.shape[0]
    res = np.ones((s, m))
    for i in range(m):
        for j in range(d):
            _sum = np.zeros((s))
            for k in range(n_c):
                _X = (k + 1.0) * x[:,j] * th_b[i,j,k] + th_c[i,j,k]
                _sum += th_a[i,j,k] * np.sin(2. * np.pi * _X) / (k + 1.0)**alpha
            res[:,i] *= _sum

        res[:,i] = res[:,i] - res[:,i].min()
        res[:,i] /= res[:,i].max()

    return res


def randman_dataset_array(m, n_classes, n_samples, d, n_cutoff,
                          alpha, seed_mf=None, seed_smp=None):

    n_samples -= n_samples % n_classes

    n_samples_per_class = int(n_samples / n_classes)

    np.random.seed(seed_mf)

    th_a = np.random.rand(n_classes, m, d, n_cutoff)
    th_b = np.random.rand(n_classes, m, d, n_cutoff)
    th_c = np.random.rand(n_classes, m, d, n_cutoff)

    np.random.seed(seed_smp)

    x = np.random.rand(n_samples, d)

    cl = np.repeat(np.arange(n_classes), n_samples_per_class)

    Y = np.zeros((n_samples, n_classes))
    Y[range(n_samples), cl] = 1.

    X = np.ndarray((n_samples, m))

    for k in range(n_classes):
        X[k * n_samples_per_class:(k + 1) * n_samples_per_class] = f(x[k * n_samples_per_class:(k + 1) * n_samples_per_class],
                                                                     th_a[k], th_b[k], th_c[k], alpha, m, d, n_cutoff)

    ind_perm = np.random.permutation(np.arange(n_samples))

    X = np.array(X[ind_perm])
    Y = np.array(Y[ind_perm])

    return X, Y
