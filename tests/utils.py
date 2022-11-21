"""
Some tools for running test simulations.
"""
import numpy as np


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

    d_targ = data_targ.shape[1]
    d_readout = data_readout.shape[1]

    assert d_targ == d_readout, 'Error: Dimensionality of target data does not match dimensionality of readout data.'

    d = d_targ

    # sort the data by increasing time (in case it is not sorted already)

    id_sort_targ = np.argsort(t_ax_targ)

    data_targ = np.array(data_targ[id_sort_targ])
    t_ax_targ = np.array(t_ax_targ[id_sort_targ])

    id_sort_readout = np.argsort(t_ax_readout)

    data_readout = np.array(data_readout[id_sort_readout])
    t_ax_readout = np.array(t_ax_readout[id_sort_readout])

    t_ax_readout_comp = t_ax_targ[:-1] + perc_readout_targ_change * (t_ax_targ[1:] - t_ax_targ[:-1])

    n_readout_comp = t_ax_readout_comp.shape[0]

    data_readout_comp = np.ndarray((n_readout_comp, d))

    for k in range(d):
        data_readout_comp[:, k] = np.interp(
            t_ax_readout_comp,
            t_ax_readout, data_readout[:, k]
        )

    loss = loss_func(data_readout_comp, data_targ[:-1])

    return loss
