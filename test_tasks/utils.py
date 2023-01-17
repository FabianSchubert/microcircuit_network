"""
Some tools for running test simulations.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_spike_times(sp, ax=None, color='k'):

    _ax = ax if ax else plt.gca()

    _ax.plot(sp[0], sp[1], '.', c=color, markersize=1)


def squ_loss(x, y):
    """
    Mean squared error loss.
    """
    return ((x - y) ** 2.).mean()


def calc_loss_interp(t_ax_targ, t_ax_readout,
                     data_targ, data_readout,
                     perc_readout_targ_change=0.9,
                     loss_func=squ_loss):
    """
    When a new target is presented to the network, it takes a certain
    amount of time steps for the network to settle into its prediction
    state. Therefore, the loss between the target and the prediction
    should be calculated not directly after the new target is presented,
    but just before the next one. This is controlled by the perc_readout_targ_change
    parameter: It sets
    (t_readout - t_targ_present_prev)/(t_targ_present_next - targ_present_prev).
    The standard value is 0.9, i.e. 90% of the time between two consecutive changes
    in the target has passed when the loss between target and prediction
    is calculated.

    The readout data must be temporally ordered, i.e. t_ax_readout must
    be increasing.
    """

    d_targ = data_targ.shape[-1]
    d_readout = data_readout.shape[-1]

    assert d_targ == d_readout, \
        '''Error: Dimensionality of target data vectors do
        not match dimensionality of readout data vectors.'''

    assert data_targ.ndim == data_readout.ndim, \
        '''Error: target data array does not have the
        same number of dimensions as readout data array.
        '''

    if data_targ.ndim == 3:
        assert data_targ.shape[1] == data_readout.shape[1], \
            '''Error: batch sizes do not match.'''

    d = d_targ
    n_batch = (data_targ.shape[1] if data_targ.ndim == 3 else 1)

    # sort the data by increasing time (in case it is not sorted already)

    id_sort_targ = np.argsort(t_ax_targ)

    data_targ = np.array(data_targ[id_sort_targ])
    t_ax_targ = np.array(t_ax_targ[id_sort_targ])

    id_sort_readout = np.argsort(t_ax_readout)

    data_readout = np.array(data_readout[id_sort_readout])
    t_ax_readout = np.array(t_ax_readout[id_sort_readout])

    t_ax_readout_comp = t_ax_targ[:-1] + perc_readout_targ_change * (t_ax_targ[1:] - t_ax_targ[:-1])

    n_readout_comp = t_ax_readout_comp.shape[0]

    data_readout_comp = np.ndarray((n_readout_comp, n_batch, d))

    for k in range(n_batch):
        for l in range(d):
            data_readout_comp[:, k, l] = np.interp(
                t_ax_readout_comp,
                t_ax_readout, data_readout[:, k, l]
            )

    loss = loss_func(data_readout_comp, data_targ[:-1])

    return loss
