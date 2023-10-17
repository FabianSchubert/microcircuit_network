import numpy as np

from scipy.ndimage import gaussian_filter

def mask(x, x0, x1):
    return 1.*(x >= x0)*(x < x1)

base_funcs = [
    lambda x: np.abs(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    lambda x: -np.abs(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    lambda x: np.sin(x*2.*np.pi) * mask(x, 0., 1.),
    lambda x: -np.sin(x*2.*np.pi) * mask(x, 0., 1.),
    lambda x: np.sqrt(np.maximum(0.,0.25-(x-0.5)**2.)) * mask(x, 0., 1.),
    lambda x: -np.sqrt(np.maximum(0.,0.25-(x-0.5)**2.)) * mask(x, 0., 1.),
    lambda x: 5.*x*mask(x,0.,0.2) + (1.-(x-.2)*10./3.)*mask(x,0.2,0.8) + (5.*(x-.8) - 1.)*mask(x,0.8,1.),
    lambda x: -(5.*x*mask(x,0.,0.2) + (1.-(x-.2)*10./3.)*mask(x,0.2,0.8) + (5.*(x-.8) - 1.)*mask(x,0.8,1.)),
    lambda x: 5.*x*mask(x,0.,0.2) + mask(x,0.2,0.8) + (1.-5.*(x-.8))*mask(x,0.8,1.),
    lambda x: -(5.*x*mask(x,0.,0.2) + mask(x,0.2,0.8) + (1.-5.*(x-.8))*mask(x,0.8,1.))
]

def mnist1d_dataset_array(n_samples, t=5., nt=100,
                          perc_max_shift = 1.0,
                          sigm_uncorr_noise = 0.1,
                          sigm_corr_noise = 0.1,
                          width_corr_noise = 0.1,
                          seed=50):

    np.random.seed(seed)

    labels = np.repeat(np.arange(10), 1 + n_samples//10)
    np.random.shuffle(labels)
    labels = labels[:n_samples]

    X = np.ndarray((n_samples, nt))
    Y = np.zeros((n_samples, nt, 11))

    t_ax = np.linspace(0.,t, nt, endpoint=False)
    dt = t_ax[1] - t_ax[0]

    for k in range(n_samples):
        shift = np.random.rand() * perc_max_shift * (t-1.)
        X[k] = base_funcs[labels[k]](t_ax-shift)

        uncorr_noise = np.random.normal(0.,1.,(nt))
        uncorr_noise = (uncorr_noise - uncorr_noise.mean()) * sigm_uncorr_noise / uncorr_noise.std()
        corr_noise = gaussian_filter(np.random.normal(0.,1.,(nt)), sigma=width_corr_noise/dt, mode="constant", cval=0.)
        corr_noise = (corr_noise - corr_noise.mean()) * sigm_corr_noise / corr_noise.std()

        X[k] += uncorr_noise + corr_noise

        Y[k,:,labels[k]] = mask(t_ax - shift, 0., 1.)
        Y[k,:,10] = 1.-Y[k,:,labels[k]]

    return X, Y
