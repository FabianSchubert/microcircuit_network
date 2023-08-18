"""
unit models
"""
import numpy as np

import unittest

import numpy.random

from mc.network import Network

from genn_models import test_model, cs_sources

import torch

from torchvision.datasets import MNIST

mnist_dataset_train = MNIST(root="./mnist", train=True, download=True)
mnist_dataset_test = MNIST(root="./mnist", train=False, download=True)

train_input = np.reshape(mnist_dataset_train.data.numpy(), (60000, 784)).flatten()
train_output = torch.nn.functional.one_hot(mnist_dataset_train.targets, num_classes=10).numpy().flatten()

test_input = np.reshape(mnist_dataset_test.data.numpy(), (10000, 784)).flatten()
test_output = torch.nn.functional.one_hot(mnist_dataset_test.targets, num_classes=10).numpy().flatten()

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
            "ext_data_pop_vars": None,
            "readout_neur_pop_vars": None
        }] * N_VAL

        self.assertIsInstance(net, Network)

        readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN, None, None, None,
            t_sign_validation = T_SIGN_VAL,
            data_validation = data_validation
        )

    def test_run_sim_val_ext_inp(self):

        T_RUN = 10000

        net = Network("testnet", test_model,
                      50, [40], 30,  # network size
                      0,  # maximum number of input arrays for external input
                      T_RUN,  # spike buffer size
                      0,  # validation spike buffer size
                      [],  # spike buffer populations
                      [],  # validation spike buffer populations
                      cs_in_init={
                          "model": cs_sources.test_source,
                          "params": {"amplitude": 1.0},
                          "vars": {},
                          "extra_global_params": {}
                      },
                      cs_out_init={
                          "model": cs_sources.test_source,
                          "params": {"amplitude": 1.0},
                          "vars": {},
                          "extra_global_params": {}
                      },
                      cs_in_init_static_twin={
                          "model": cs_sources.test_source,
                          "params": {"amplitude": 1.0},
                          "vars": {},
                          "extra_global_params": {}
                      },
                      cs_out_init_static_twin={
                          "model": cs_sources.test_source,
                          "params": {"amplitude": 1.0},
                          "vars": {},
                          "extra_global_params": {}
                      }
                      )

        T_SIGN_VAL = np.linspace(0., T_RUN, 10)
        N_VAL = T_SIGN_VAL.shape[0]

        data_validation = [{
            "T": 1000,
            "ext_data_pop_vars": None,
            "readout_neur_pop_vars": None
        }] * N_VAL

        self.assertIsInstance(net, Network)

        readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN, None, None, None,
            t_sign_validation=T_SIGN_VAL,
            data_validation=data_validation
        )

    def test_run_sim_val_mnist(self):
        T_RUN = 10000
        T_RUN_TEST = 3000

        rng = numpy.random.default_rng()

        N_BATCH = 10

        N_EPOCHS = 2
        N_TRAIN = 60000
        N_INP_TIMES = int(N_EPOCHS * N_TRAIN / N_BATCH)

        INPUT_IDS_TRAIN = np.concatenate([rng.permutation(N_TRAIN) for _ in range(N_EPOCHS)])
        INPUT_TIMES_TRAIN = np.arange(N_INP_TIMES) * T_RUN / N_INP_TIMES

        N_TEST = 10000
        N_EPOCHS_TEST = 1
        N_INP_TIMES_TEST = int(N_EPOCHS_TEST * N_TEST / N_BATCH)

        INPUT_IDS_TEST = np.concatenate([rng.permutation(N_TEST) for _ in range(N_EPOCHS_TEST)])
        INPUT_TIMES_TEST = np.arange(N_INP_TIMES_TEST) * T_RUN_TEST / N_INP_TIMES_TEST

        cs_in = {
                  "model": cs_sources.step_source_model,
                  "params": {"n_samples_set": N_TRAIN,
                             "pop_size": 784,
                             "batch_size": N_BATCH,
                             "input_id_list_size": N_EPOCHS * N_TRAIN,
                             "input_times_list_size": N_INP_TIMES},
                  "vars": {"idx": 0,
                           "t_next": 0.0},
                  "extra_global_params": {"data": train_input,
                                          "input_id_list": INPUT_IDS_TRAIN,
                                          "input_times_list": INPUT_TIMES_TRAIN}
                }

        cs_out = {
            "model": cs_sources.step_source_model,
            "params": {"n_samples_set": N_TRAIN,
                       "pop_size": 10,
                       "batch_size": N_BATCH,
                       "input_id_list_size": N_EPOCHS * N_TRAIN,
                       "input_times_list_size": N_INP_TIMES},
            "vars": {"idx": 0,
                     "t_next": 0.0},
            "extra_global_params": {"data": train_output,
                                    "input_id_list": INPUT_IDS_TRAIN,
                                    "input_times_list": INPUT_TIMES_TRAIN}
        }

        cs_in_test = {
            "model": cs_sources.step_source_model,
            "params": {"n_samples_set": N_TEST,
                       "pop_size": 784,
                       "batch_size": N_BATCH,
                       "input_id_list_size": N_EPOCHS_TEST * N_TEST,
                       "input_times_list_size": N_INP_TIMES_TEST},
            "vars": {"idx": 0,
                     "t_next": 0.0},
            "extra_global_params": {"data": test_input,
                                    "input_id_list": INPUT_IDS_TEST,
                                    "input_times_list": INPUT_TIMES_TEST}
        }

        cs_out_test = {
            "model": cs_sources.step_source_model,
            "params": {"n_samples_set": N_TEST,
                       "pop_size": 10,
                       "batch_size": N_BATCH,
                       "input_id_list_size": N_EPOCHS_TEST * N_TEST,
                       "input_times_list_size": N_INP_TIMES_TEST},
            "vars": {"idx": 0,
                     "t_next": 0.0},
            "extra_global_params": {"data": test_output,
                                    "input_id_list": INPUT_IDS_TEST,
                                    "input_times_list": INPUT_TIMES_TEST}
        }


        net = Network("testnet", test_model,
                      784, [100], 10,  # network size
                      0,  # maximum number of input arrays for external input
                      T_RUN,  # spike buffer size
                      0,  # validation spike buffer size
                      [],  # spike buffer populations
                      [],  # validation spike buffer populations
                      n_batches=N_BATCH,
                      n_batches_val=N_BATCH,
                      cs_in_init=cs_in,
                      cs_out_init=cs_out,
                      cs_in_init_static_twin=cs_in_test,
                      cs_out_init_static_twin=cs_out_test
                      )

        T_SIGN_VAL = np.linspace(0., T_RUN, 10)
        N_VAL = T_SIGN_VAL.shape[0]

        data_validation = [{
            "T": T_RUN_TEST,
            "ext_data_pop_vars": None,
            "readout_neur_pop_vars": None
        }] * N_VAL

        self.assertIsInstance(net, Network)

        readout_neur_arrays, readout_syn_arrays, readout_spikes, results_validation = net.run_sim(
            T_RUN, None, None, None,
            t_sign_validation=T_SIGN_VAL,
            data_validation=data_validation)

if __name__ == '__main__':
    unittest.main()