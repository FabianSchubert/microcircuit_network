"""
unit tests
"""
import numpy as np

import unittest

from mc.network import Network

from . import test_model


class TestNetwork(unittest.TestCase):

    def test_init_net(self):

        net = Network("testnet", test_model,
                      10, [10], 10,
                      10,
                      1000,
                      1000,
                      [],
                      [])

        self.assertIsInstance(net, Network)

    def test_run_sim_no_val_no_ext_inp(self):

        T_RUN = 10000

        net = Network("testnet", test_model,
                      50, [40], 30, # network size
                      0, # maximum number of input arrays for external input
                      T_RUN, # spike buffer size
                      0, # validation spike buffer size
                      [], # spike buffer populations
                      []) # validation spike buffer populations

        self.assertIsInstance(net, Network)

        readout_neur_arrays, readout_syn_arrays, readout_spikes = net.run_sim(
            T_RUN, None, None, None, None, None, None
        )

    def test_run_sim_val_no_ext_inp(self):

        """
        T,
                t_sign,
                ext_data_input, ext_data_output,
                ext_data_pop_vars, readout_neur_pop_vars,
                readout_syn_pop_vars,
                t_sign_validation=None,
                data_validation=None,
                show_progress=True,
                show_progress_val=True
        """

        T_RUN = 10000

        net = Network("testnet", test_model,
                      50, [40], 30,  # network size
                      0,  # maximum number of input arrays for external input
                      T_RUN,  # spike buffer size
                      0,  # validation spike buffer size
                      [],  # spike buffer populations
                      [])  # validation spike buffer populations

        T_SIGN_VAL = np.arange(T_RUN)[::1000]
        N_VAL = T_SIGN_VAL.shape[0]

        data_validation = [{
            "T": 1000,
            "t_sign": None,
            "ext_data_input": None,
            "ext_data_pop_vars": None,
            "readout_neur_pop_vars": None
        }] * N_VAL

        self.assertIsInstance(net, Network)

        readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN, None, None, None, None, None, None,
            t_sign_validation = T_SIGN_VAL,
            data_validation = data_validation
        )

if __name__ == '__main__':
    unittest.main()